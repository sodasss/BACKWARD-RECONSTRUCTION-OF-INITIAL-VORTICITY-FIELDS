clear; clc; close all;

%% -------------------------------
%  Parameters
% -------------------------------

N  = 64;    
L  = 1.0;
h  = L / N;

nu = 0.01;
T  = 5.0;            % final time per trajectory
m  = 50;             % save 50 frames
n_traj = 2000;       % number of trajectories

%% Time step
C_adv = 0.2; 
C_dif = 0.2;

[xc, yc] = meshgrid(h/2:h:1-h/2, h/2:h:1-h/2);

%% Velocity (fixed)
u =  pi * cos(pi*yc) .* sin(pi*xc);
v = -pi * sin(pi*yc) .* cos(pi*xc);
Umax = max(abs(u(:)) + abs(v(:)));

dt_adv = C_adv * h / Umax;
dt_dif = C_dif * h^2 / nu;
dt = min(dt_adv, dt_dif);

Nt = ceil(T / dt);
dt = T / Nt;

fprintf("Using dt = %.3e, Nt=%d\n", dt, Nt);

%% Allocate dataset
data = zeros(N, N, m, n_traj);   % [64,64,50,2000]
initials = zeros(N, N, n_traj);

%% -------------------------------
%  Loop over trajectories
% -------------------------------
for k = 1:n_traj

    % ===== Random initial condition =====
    type = randi(3);
    switch type
        case 1
            % Gaussian blob
            cx = 0.2 + 0.6*rand();
            cy = 0.2 + 0.6*rand();
            sigma = 0.05 + 0.1*rand();
            w = exp(-((xc-cx).^2 + (yc-cy).^2) / sigma);
        case 2
            % Multiple Gaussians
            w = zeros(N,N);
            for g = 1:2+randi(2)
                cx = rand();
                cy = rand();
                sg = 0.03 + 0.08*rand();
                w = w + 0.8*rand()*exp(-((xc-cx).^2 + (yc-cy).^2)/sg);
            end
        case 3
            % Random Fourier modes
            w = sin(2*pi*xc).*sin(2*pi*yc) ...
              + 0.5*sin(4*pi*xc).*cos(3*pi*yc) ...
              + 0.3*cos(5*pi*xc).*sin(2*pi*yc);
            w = w + 0.1*randn(N,N);
    end

    initials(:,:,k) = w;  % save initial state

    %% Alloc for this trajectory
    traj = zeros(N, N, m);
    save_count = 1;

    %% Time stepping
    for n = 1:Nt
        t = n * dt;

        % forcing
        f1 = sin(2*pi*xc).*sin(2*pi*yc);
        f2 = 0.5*sin(6*pi*xc).*sin(4*pi*yc);
        f3 = 10 * exp(-200*((xc-0.25).^2 + (yc-0.75).^2));
        time_factor = cos(2*pi*t/T);
        f = time_factor .* (f1 + f2) + f3;

        % derivatives
        wx = (w(:,[2:end end]) - w(:,[1 1:end-1])) / (2*h);
        wy = (w([2:end end],:) - w([1 1:end-1],:)) / (2*h);
        adv = u.*wx + v.*wy;

        Lap = ( ...
            w([2:end end],:) + w([1 1:end-1],:) + ...
            w(:,[2:end end]) + w(:,[1 1:end-1]) - ...
            4*w ) / h^2;

        w = w + dt*(-adv + nu*Lap + f);

        % boundary
        w([1 end],:) = 0;
        w(:,[1 end]) = 0;

        % Save evenly spaced m frames
        if n == round((save_count/m)*Nt)
            traj(:,:,save_count) = w;
            save_count = save_count + 1;
        end
    end

    data(:,:,:,k) = traj;

    if mod(k,100)==0
        fprintf("Finished %d / %d trajectories\n", k, n_traj);
    end
end

%% -------------------------------
%  Save to MAT file
% -------------------------------
save("w_data.mat","data","initials","-v7.3");

%% -------------------------------
%  Save to HDF5 (recommended)
% -------------------------------
h5create("w_data.h5","/data",size(data));
h5write("w_data.h5","/data",data);

h5create("w_data.h5","/initials",size(initials));
h5write("w_data.h5","/initials",initials);
