%% pf3d_step3_singlefile.m
% 3-DOF R-R-R robot — Potential-Fields (3D, class style) in ONE file.
% What this implements (exactly like lectures):
%   • Define attractive/repulsive "workspace" forces at EACH frame origin o1,o2,o3
%   • Map each force to joint torques with the ORIGIN linear Jacobians (J_o_i^T * F_i)
%   • SUM torques in joint space, then gradient-descent step in q
%   • Plot robot at start/end and the o3 path; print diagnostics while iterating
%
% IMPORTANT: "Workspace forces" here are ARTIFICIAL (from potentials), not physical
% dynamics (no mass/inertia/gravity). We are doing kinematic planning, not simulating
% Newton-Euler. Gravity etc. would belong to a dynamics controller; not needed here.

clear; clc; close all;

%% --------------------------- 0) Workspace build ---------------------------
WS = build_workspace();     % Your DH, joint limits, belts (for visualization), obstacles

%% --------------------------- 1) Parameters --------------------------------
P = struct();
% Tuning knobs (per lecture)
P.zeta  = [1.0; 0.6; 0.8];     % attractive strength for o1,o2,o3 (make one "leader", e.g., o3)
P.dstar = 0.12;                % attractive switch distance (cone <-> parabola)
P.eta   = [120; 120; 120];     % repulsive strength for o1,o2,o3 (spherical obstacles)
P.rho0  = 0.30;                % repulsive influence range (meters)
% Descent and guards
P.alpha = 0.03;                % joint step (rad/iter)
P.eps   = 5e-3;                % stop if ||q - q_goal|| < eps
P.delta = 5e-3;                % clamp distances >= delta to avoid 1/rho blow-ups
P.maxit = 4000;
P.q_lim = WS.q_lim;

% Choose mode: attractive-only (for quick testing) or full (att + rep)
use_repulsion = true;          % set false to test attraction first

%% --------------------------- 2) Goal origins ------------------------------
% Class method: each origin o_i is pulled toward its counterpart at q_goal.
[o_goal, ~, ~] = fk_3R_world(WS.DH, WS.q_goal);   % 3x3 matrix: [o1g o2g o3g]

%% --------------------------- 3) Planner loop ------------------------------
q = WS.q_start;            % start configuration (joint space)
o3_path = [];              % tip path
Ulog = [];                 % total potential per iter
minR = [];                 % min clearance per iter
stalled = false;

% NEW: logs for forces & torque
FattLog = [];              % 3 x T  (|F_att| at o1,o2,o3)
FrepLog = [];              % 3 x T  (|F_rep| at o1,o2,o3)
TauLog  = [];              % 1 x T  (||tau||)
FtotPath = [];             % 3 x T  (resultant task-space force we’ll quiver at o3)

fprintf('PF-3D planner (class-style). Repulsion: %d\n', use_repulsion);

for k = 1:P.maxit
    % --- FK: get world origins {o1,o2,o3} and z-axes for Jacobians ---
    [o, z, ~] = fk_3R_world(WS.DH, q);
    o1=o(:,1); o2=o(:,2); o3=o(:,3);

    % --- Attractive (hybrid) forces & potentials at each origin ---
    [F1a, U1a] = F_attractive_hybrid(o1, o_goal(:,1), P.zeta(1), P.dstar);
    [F2a, U2a] = F_attractive_hybrid(o2, o_goal(:,2), P.zeta(2), P.dstar);
    [F3a, U3a] = F_attractive_hybrid(o3, o_goal(:,3), P.zeta(3), P.dstar);

    % --- Repulsive (spheres) forces if enabled (sum over all spheres) ---
    if use_repulsion && ~isempty(WS.Obs)
        [F1r, U1r, r1] = F_repulsive_spheres(o1, WS.Obs, P.eta(1), P.rho0, P.delta);
        [F2r, U2r, r2] = F_repulsive_spheres(o2, WS.Obs, P.eta(2), P.rho0, P.delta);
        [F3r, U3r, r3] = F_repulsive_spheres(o3, WS.Obs, P.eta(3), P.rho0, P.delta);
    else
        F1r=zeros(3,1); U1r=0; r1=inf;
        F2r=zeros(3,1); U2r=0; r2=inf;
        F3r=zeros(3,1); U3r=0; r3=inf;
    end

    % === DIAGNOSTICS (before mapping to joint torques) =======================
    % magnitudes of attractive & repulsive forces at each origin
    nF1a = norm(F1a);  nF2a = norm(F2a);  nF3a = norm(F3a);
    nF1r = norm(F1r);  nF2r = norm(F2r);  nF3r = norm(F3r);

    % resultant task-space force to visualize along the path (at o3)
    Ftot = (F3a + F3r);


    % --- Origin Jacobians (linear), map forces -> torques, SUM in joint space ---
    J1 = jacobian_origin_3R(o, z, 1);
    J2 = jacobian_origin_3R(o, z, 2);
    J3 = jacobian_origin_3R(o, z, 3);
    tau = J1.'*(F1a+F1r) + J2.'*(F2a+F2r) + J3.'*(F3a+F3r)

    % --- Gradient-descent step in configuration space (normalize tau) ---
    nTau = norm(tau);
    if nTau < 1e-12
        warning('Near-zero torque at it=%d (minima or singular). Tune params or change seed.', k);
        stalled = true; break;
    end
    q = q + P.alpha * (tau / nTau);
    q = clamp_limits(q, P.q_lim);

    % --- Logs / prints -----------------------------------------------------
    o3_path(:,end+1) = o3;                         %#ok<AGROW>
    Utot = (U1a+U2a+U3a) + (U1r+U2r+U3r);
    Ulog(end+1) = Utot;                            %#ok<AGROW>
    minR(end+1) = min([r1 r2 r3]);                 %#ok<AGROW>

    % push per-iteration diagnostics
    FattLog(:,end+1) = [nF1a; nF2a; nF3a];
    FrepLog(:,end+1) = [nF1r; nF2r; nF3r];
    TauLog(end+1)    = nTau;
    FtotPath(:,end+1)= Ftot;

    
    if mod(k,100)==0
        fprintf(['it%4d | ||q-qg||=%.4f  U=%.2e  minρ=%.3f  |τ|=%.2e  ', ...
             '|Fatt|=[%.2f %.2f %.2f]  |Frep|=[%.2f %.2f %.2f]\n'], ...
            k, norm(q - WS.q_goal), Utot, minR(end), nTau, ...
            nF1a, nF2a, nF3a, nF1r, nF2r, nF3r);
    end


    if norm(q - WS.q_goal) < P.eps, break; end

    % --- Simple stall detector: no improvement over a window --------------
    win=200;
    if k>win && norm(o3_path(:,end)-o3_path(:,end-win)) < 1e-4
        warning('Stagnation (o3 not moving). Try smaller d*, larger rho0, bigger eta, or smaller alpha.');
        stalled = true; break;
    end
end

fprintf('Done. iters=%d  ||q-qg||=%.4g  stalled=%d\n', k, norm(q-WS.q_goal), stalled);













%% --------------------------- 4) Plots ------------------------------------
plot_workspace3D(WS, WS.q_start, q, WS.Obs, o3_path, FtotPath, ...
    sprintf('PF (3D): %s', ternary(use_repulsion,'Att+Rep','Att-only')));


figure('Name','Convergence'); 
subplot(2,1,1); plot(Ulog,'LineWidth',1.4); grid on; ylabel('U total');
subplot(2,1,2); plot(minR,'LineWidth',1.4); grid on; ylabel('min \\rho'); xlabel('iteration');



figure('Name','Forces & Torque'); 
t = 1:numel(TauLog);
subplot(3,1,1); plot(t, FattLog, 'LineWidth',1.3); grid on;
ylabel('|F_{att}|'); legend('o1','o2','o3','Location','best');
subplot(3,1,2); plot(t, FrepLog, 'LineWidth',1.3); grid on;
ylabel('|F_{rep}|'); legend('o1','o2','o3','Location','best');
subplot(3,1,3); plot(t, TauLog, 'LineWidth',1.6); grid on;
ylabel('||\tau||'); xlabel('iteration');


%% ========================== LOCAL FUNCTIONS ===============================

function WS = build_workspace()
% Base frame at robot base: x right, y forward (toward main belt), z up.

% --- YOUR DH ---
WS.DH = [ 0.00,  0.35,  +pi/2,  0;   % link1
          0.70,  0.00,   0.00,  0;   % link2
          0.55,  0.00,   0.00,  0 ]; % link3

% Joint limits
WS.q_lim = [deg2rad(-170) deg2rad(170);
            deg2rad(-150) deg2rad(150);
            deg2rad(-150) deg2rad(150)];

% Start/Goal seeds (start up/left; goal forward/lower)
WS.q_start = deg2rad([  90;  45; -45]);   % top-level belt side pose
WS.q_goal  = deg2rad([  0;  15;  -45]);   % forward, a bit lower in z

% --- Belts (for visualization only) ---
% Top belt (blue): move to y = +1 (same z; extend ~1 m along x)
WS.areaA.z_min = 0.4475;
WS.areaA.x_rng = [-0.50, 0.50];   % starts at origin and runs to x=+0.5 (edit as you like)
WS.areaA.y_val = 1.0;             % << moved from ~0 to +1.0


% Main belt (green): along y at x = 1.0, z >= 0.149 m
WS.areaB.z_min = 0.1490;
WS.areaB.x_val = 1.0;        % << fixed x position
WS.areaB.y_rng = [0.0, 1.0]; % << starts at y=0, ends at y=1


% Obstacles (start empty; add when testing repulsion)
% Obstacles (spheres)
WS.Obs = struct('c', {}, 'r', {});   % start empty
% --- ADD THIS SPHERE ---
WS.Obs(1).c = [0.75; 0.9; 0.6];   % center [x;y;z] in meters
WS.Obs(1).r = 0.25;               % radius in meters

WS.Obs(2).c = [0.23; 1; 0.7];   % center [x;y;z] in meters
WS.Obs(2).r = 0.17;               % radius in meters

WS.Obs(3).c = [1; 0.4; 0.25];   % center [x;y;z] in meters
WS.Obs(3).r = 0.17;               % radius in meters


WS.Obs(4).c = [0.25; 0.9; 1.1];   % center [x;y;z] in meters
WS.Obs(4).r = 0.25;               % radius in meters



% e.g.:
% WS.Obs(1).c=[-0.25; 0.20; 0.30]; WS.Obs(1).r=0.09;
% WS.Obs(2).c=[ 0.10; 0.55; 0.22]; WS.Obs(2).r=0.08;
end


function [o_all, z_all, T_all] = fk_3R_world(DH, q)
% Forward kinematics (standard DH) to WORLD. Returns:
%   o_all = [o1 o2 o3], z_all = [z1 z2 z3], T_all(:,:,i) = T_0^i

T = eye(4);
o_all = zeros(3,3); 
z_all=zeros(3,3); 
T_all=zeros(4,4,3);

for i=1:3

    a=DH(i,1); d=DH(i,2); alpha=DH(i,3); theta=q(i);

    A = trotz(theta) * transl([0,0,d]) * trotx(alpha) * transl([a,0,0]);
    T = T * A;
    o_all(:,i) = T(1:3,4);
    z_all(:,i) = T(1:3,3);
    T_all(:,:,i)=T;
end
end

function J = jacobian_origin_3R(o_all, z_all, i)
% Linear Jacobian of ORIGIN o_i wrt revolute joints 1..3:
% column j = z_{j-1} × (o_i - o_{j-1}), all in WORLD.
J=zeros(3,3); o0=[0;0;0]; z0=[0;0;1]; oi=o_all(:,i);
for j=1:3

    if j==1, oj=o0; zj=z0; 
    else, oj=o_all(:,j-1); 
        zj=z_all(:,j-1); 
    end
    J(:,j) = cross(zj, (oi - oj));
end
end

function [F,U] = F_attractive_hybrid(o, og, zeta, dstar)
% Hybrid attractive field (lecture):
% Parabolic near goal (smooth landing), Conic far away (constant pull).
dvec = o - og; 
d = norm(dvec) + eps;

if d <= dstar
    U = 0.5*zeta*d^2;
    F = -zeta * dvec;
else
    U = zeta*dstar*(d - 0.5*dstar);
    F = -zeta*dstar * (dvec / d);
end
end

function [Fsum, Usum, rhoMin] = F_repulsive_spheres(o, Obs, eta, rho0, delta)
% Sum of repulsive forces from a set of SPHERES acting on point o
% U = 0.5*eta*(1/rho - 1/rho0)^2 for rho<rho0, else 0  ;  rho = distance to surface
Fsum=zeros(3,1); Usum=0; rhoMin=inf;
for k=1:numel(Obs)
    c=Obs(k).c; R=Obs(k).r;
    v = o - c;
    L = norm(v) + eps;
    rho = max(L - R, delta);          % distance to surface (clamped)
    rhoMin = min(rhoMin, rho);
    if rho < rho0
        grad_rho = v / L;             % ∂rho/∂o  (outward normal)
        Usum = Usum + 0.5*eta*(1/rho - 1/rho0)^2;
        Fsum = Fsum + eta*(1/rho - 1/rho0)*(1/rho^2) * grad_rho;
    end
end
end

function q = clamp_limits(q, qlim)
for i=1:numel(q)
    q(i) = min(max(q(i), qlim(i,1)), qlim(i,2));
end
end


function plot_workspace3D(WS, q_start, q_end, Obs, o3_traj, FtotPath, ttl)

figure('Name','Workspace 3D'); hold on; axis equal; grid on;
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]'); view(-135, 25);   % was view(45,25)% 
title(ttl);

% Top belt (blue): along x at y = yA, z >= zA
xA=WS.areaA.x_rng; 
yA=WS.areaA.y_val+[-0.05 0.05]; 
zA=WS.areaA.z_min+[0 0.01];

drawBox([xA(1) yA(1) zA(1)], [xA(2)-xA(1), yA(2)-yA(1), zA(2)-zA(1)], [0.2 0.4 1], 0.15);

% Main belt (green): along y at x = xB, z >= zB
xB = WS.areaB.x_val + [-0.05 0.05];      % thin slab around x=1.0
yB = WS.areaB.y_rng;                      % from y=0 to y=1
zB = WS.areaB.z_min + [0 0.01];           % small thickness in z

drawBox([xB(1) yB(1) zB(1)], ...
        [xB(2)-xB(1), yB(2)-yB(1), zB(2)-zB(1)], ...
        [0.1 0.8 0.3], 0.15);



% Obstacles (spheres)
for i=1:numel(Obs), drawSphere(Obs(i).c, Obs(i).r, 0.18); end

% Robot at start / end
drawRobot(WS.DH, q_start, [0.0 0.6 0.0]);   % green
drawRobot(WS.DH, q_end,   [0.8 0.0 0.0]);   % red

% o3 path
if ~isempty(o3_traj)
    plot3(o3_traj(1,:), o3_traj(2,:), o3_traj(3,:), 'k-', 'LineWidth', 1.8);
end

% --- quiver the resultant task-space force at o3 along the path -----------
if ~isempty(o3_traj) && ~isempty(FtotPath)
    P = o3_traj;  Fp = FtotPath;
    skip = max(1, floor(size(P,2)/25));  % ~25 arrows max
    quiver3(P(1,1:skip:end), P(2,1:skip:end), P(3,1:skip:end), ...
            Fp(1,1:skip:end), Fp(2,1:skip:end), Fp(3,1:skip:end), ...
            0.2, 'k');   % 0.2 = arrow scale
end


legend('Top belt','Main belt','start cfg','end cfg','o3 path');
end

% --- draw helpers
function drawRobot(DH,q,color)
[o,~,~]=fk_3R_world(DH,q);
p=[0 0 0]'; 
plot3(p(1),p(2),p(3),'.','Color',color,'MarkerSize',16);

for i=1:3
    plot3([p(1) o(1,i)], [p(2) o(2,i)], [p(3) o(3,i)], '-', 'Color',color, 'LineWidth',2);
    p=o(:,i);
end
end
function drawSphere(c,r,alp)
[X,Y,Z]=sphere(32); surf(c(1)+r*X,c(2)+r*Y,c(3)+r*Z,'FaceAlpha',alp,'EdgeColor','none','FaceColor',[1 0 0]);
end
function drawBox(origin, dims, color, alp)
[x0,y0,z0]=deal(origin(1),origin(2),origin(3)); [dx,dy,dz]=deal(dims(1),dims(2),dims(3));
[X,Y,Z] = ndgrid([x0 x0+dx],[y0 y0+dy],[z0 z0+dz]);
patchSurf([X(1,1,1) X(1,2,1) X(2,2,1) X(2,1,1)], [Y(1,1,1) Y(1,2,1) Y(2,2,1) Y(2,1,1)], z0*ones(1,4), color, alp);
patchSurf([X(1,1,2) X(1,2,2) X(2,2,2) X(2,1,2)], [Y(1,1,2) Y(1,2,2) Y(2,2,2) Y(2,1,2)], (z0+dz)*ones(1,4), color, alp);
patchSurf([x0 x0 x0 x0],[y0 y0+dy y0+dy y0],[z0 z0 z0+dz z0+dz],color,alp);
patchSurf([x0+dx x0+dx x0+dx x0+dx],[y0 y0+dy y0+dy y0],[z0 z0 z0+dz z0+dz],color,alp);
patchSurf([x0 x0+dx x0+dx x0],[y0 y0 y0 y0],[z0 z0 z0+dz z0+dz],color,alp);
patchSurf([x0 x0+dx x0+dx x0],[y0+dy y0+dy y0+dy y0+dy],[z0 z0 z0+dz z0+dz],color,alp);
end
function patchSurf(x,y,z,color,alp)
patch(x,y,z,color,'FaceAlpha',alp,'EdgeColor','none');
end

% --- tiny helpers ---
function out = ternary(cond,a,b), if cond, out=a; else, out=b; end, end

function T = trotz(th), c=cos(th); 
s=sin(th); 
T=[c -s 0 0; s c 0 0; 0 0 1 0; 0 0 0 1];
end
function T = trotx(al), c=cos(al); s=sin(al); T=[1 0 0 0; 0 c -s 0; 0 s c 0; 0 0 0 1]; end
function T = transl(v), T=eye(4); T(1:3,4)=v(:); end
