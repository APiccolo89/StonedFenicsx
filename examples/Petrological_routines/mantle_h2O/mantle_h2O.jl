using MAGEMin_C, HDF5, PyCall, Conda
using Base.Threads
@pyimport scipy.interpolate as si
#------------------------------------------------------------
function mantle_h2O(data::MAGEMin_Data, X::Array{Float64,1}, P::Float64, T::Float64)
    Xoxides = ["SiO2", "Al2O3", "MgO", "FeO", "O","H2O", "S", "CaO", "Na2O"]
    out     = single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in="wt",name_solvus=true)

    return out.frac_F_wt, out.bulk_S_wt, out.rho, out
end
#------------------------------------------------------------
function extract_data(file_path::String)
    file = h5open(file_path, "r")

    coord = read(file["/Mesh/mesh/geometry"])

    x = coord[1,:];
    y = coord[2,:];
    P = read(file["/Function/Lit Pres  [GPa]/0"]);
    T = read(file["/Function/Temperature  [degC]/0"]);
    v = read(file["/Function/Velocity  [cm/yr]/0"]);
    vx = v[1,:]*1e-5/3.15e7; # convert from cm/yr to km/s
    vy = v[2,:]*1e-5/3.15e7; # convert from cm/yr to km/s

    return x, y, P, T, vx, vy
end
#------------------------------------------------------------
function create_phase_diagram(data::MAGEMin_Data, X::Array{Float64,1},name::String,water_adjustment::Bool)
    T = 20.0:10.0:1600.0
    P = 0.1:0.1:80
    
    fl_grid = zeros(length(T), length(P))
    rho_grid = zeros(length(T), length(P))
    H2O_grid = zeros(length(T), length(P))
    
    for i in 1:length(T)
        t0 = time()
        for j in 1:length(P)
            fl_grid[i,j], X_solid ,rho_grid[i,j], out = mantle_h2O(data, X, P[j], T[i])
            H2O_grid[i,j] = X_solid[6]
        end     
        println("$(time() - t0) seconds")
    end
    
    # Save the phase diagram data to an HDF5 file
    h5open("$(name)_phase_diagram.h5", "w") do file
        write(file, "T", collect(T))
        write(file, "P", collect(P)./10)
        write(file, "fl_grid", fl_grid)
        write(file, "rho_grid", rho_grid)
        write(file, "H2O_grid", H2O_grid)
    end

end
#------------------------------------------------------------
mutable struct Points
           x::Array{Float64,1}
           y::Array{Float64,1}
           vx::Array{Float64,1}
           vy::Array{Float64,1}
           P::Array{Float64,1}
           T::Array{Float64,1}
           fl::Array{Float64,1}
           fl_ex::Array{Float64,1}
           df::Array{Float64,1}
           X::Array{Float64,2}
           H2O::Array{Float64,1}
end
#------------------------------------------------------------
function main()
    data = Initialize_MAGEMin("ume",verbose=false);
    file_path = "/Users/wlnw570/Work/Leeds/Fenics_tutorial/examples/Japan/Japan_tmax30_vc6_SelfConsistent_pr1_wzWetQuartzite_disl_0/Steady_state.h5"
    
    # Extract data from the HDF5 file
    x, y, P, T, vx, vy = extract_data(file_path)
    # Transpose the arrays to ensure they are in the correct shape for interpolation
    x = transpose(x);
    y = transpose(y);
    vx = transpose(vx);
    vy = transpose(vy);
    T = transpose(T[1:end]);
    P = transpose(P[1:end]);

    # Initialize points initial coordinate for the points
    xp = zeros(30);
    yp = zeros(30);
    Pp = zeros(30);
    Tp = zeros(30);
    # Set the initial coordinates for the points
    yp = collect(LinRange(-30.0,0.0,30));
    
    Point = Points(xp, yp,zeros(30),zeros(30), Pp, Tp, zeros(30), zeros(30), zeros(30), zeros(30,9), zeros(30));
    # Create interpolators for velocity, pressure, and temperature fields
    interp_vx = si.LinearNDInterpolator((x,y), vx)
    interp_vy = si.LinearNDInterpolator((x,y), vy)
    interp_P = si.LinearNDInterpolator((x,y), P)
    interp_T = si.LinearNDInterpolator((x,y), T)
    # Create the scaling
    sec_year = 365.25*24*3600*1e6
    # Initialise time and time step
    t = 0.0 
    dt = 0.01 * sec_year # 100 kyr in seconds
    # Set the maximum time for the simulation
    max_time = 20 * sec_year # 10 Myr in seconds
    X_mantle = [47.1100; 2.6800; 36.49000; 7.5200; 0.0000;  2.00000; 0.00000; 4.2000; 0.00000];
    X_Basalt = [49.0800; 15.2300;7.37000;10.2800;0.0000;  3.10000; 0.00000; 10.7700; 2.52000];
    X_Gabbro =  [51.01;16.23;9.27;6.25; 0.00; 0.80; 0.00; 12.60; 2.82]
    # Normalize the compositions to 100 wt%
    X_mantle .= X_mantle./sum(X_mantle).*100.0
    X_Basalt .= X_Basalt./sum(X_Basalt).*100.0
    X_Gabbro .= X_Gabbro./sum(X_Gabbro).*100.0

    println("Initialising points...")
    println("X_mantle: $(X_mantle)")
    println("X_Basalt: $(X_Basalt)")
    println("X_Gabbro: $(X_Gabbro)")
    # Set Initial composition for the points based on their initial y coordinate
    for i in 1:30
        if Point.y[i] < -7.0
            X = X_mantle
        elseif Point.y[i] >= -2.0
            X = X_Basalt
        elseif Point.y[i] >= -7.0 && Point.y[i] < -2.0
            X = X_Gabbro
        else
                error("Unexpected y coordinate: $(Point.y[i])")
        end
        Point.X[i,:] .= X
    end 

    # Set up arrays to save the results
    time = collect(LinRange(0.0, max_time, 1000))
    dt = max_time/1000
    
    x_save = zeros(30,1000);
    y_save = zeros(30,1000);
    fl_save = zeros(30,1000);
    fl_out = zeros(30,1000);
    p_save = zeros(30,1000);
    T_save = zeros(30,1000);
    H2O_save = zeros(30,1000);

    # Time-stepping loop
    for it in 1:length(time)
        # Update the positions of the points based on the velocity field 
        if t > 0.0
            Point.x[:] = Point.x[:] + Point.vx[:] * dt
            Point.y[:] = Point.y[:] + Point.vy[:] * dt
        end
        # Interpolate the pressure, temperature, and velocity fields at the new positions of the points
        for i in 1:30
            Point.P[i] = only(interp_P(Point.x[i], Point.y[i]))*1e9/1e5/1e3 # convert from GPa to bar to kbar
            Point.T[i] = only(interp_T(Point.x[i], Point.y[i])) 
            Point.vx[i] = only(interp_vx(Point.x[i], Point.y[i]))
            Point.vy[i] = only(interp_vy(Point.x[i], Point.y[i]))
            fl,X_S,rho,_ = mantle_h2O(data, Point.X[i,:], Point.P[i], Point.T[i]) # calculate the melt fraction and composition at the new position of the point
            Point.fl[i] = fl
            Point.H2O[i] = X_S[6]
            
            if i > 1
                if Point.fl[i]>0.00
                    Point.fl_ex[i] = Point.fl[i]-0.00;
                    Point.X[i,:] .= X_S[:]
                end
            end
        end
        
        x_save[:,it] .= Point.x[:]
        y_save[:,it] .= Point.y[:]
        p_save[:,it] .= Point.P[:]./10
        T_save[:,it] .= Point.T[:]
        fl_out[:,it] .= Point.fl_ex[:]
        H2O_save[:,it] .= Point.H2O[:]
        t = t + dt
        println("Time: $(t/1e6/365.25/24/3600) Myr")  
    end
    
    time =time.*1e-6/365.25/24/3600 # convert time to Myr
    
    h5open("data.h5", "w") do file
        write(file, "x", x_save)
        write(file, "y", y_save)
        write(file, "P", p_save)
        write(file, "T", T_save)
        write(file, "fl_ex", fl_out)
        write(file, "H2O", H2O_save)
        write(file,"max_time", time)
        write(file,"time", time)

    end

    # Create Phase diagram of the initial composition 
    create_phase_diagram(data, X_mantle, "Mantle", false)
    create_phase_diagram(data, X_Basalt, "Basalt", false)
    create_phase_diagram(data, X_Gabbro, "Gabbro", false)

    #create_phase_diagram(data, X_mantle, "Mantle",water_adjustment=true)
    #create_phase_diagram(data, X_Basalt, "Basalt",water_adjustment=true)
    #create_phase_diagram(data, X_Gabbro, "Gabbro",water_adjustment=true)

end
#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------
# Main Function
main()