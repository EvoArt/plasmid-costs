using GLMakie, DifferentialEquations

# model adapted from https://www.nature.com/articles/s41467-021-22849-y
# The model is written out differently from the LaTeX in the manuscript
# but it is mathematically identical. 
U(R,Vₘₐₓ,Kₘ) = (Vₘₐₓ*R)/(Kₘ+R)

function simple_levin(du,u,p,t)
    R,Bₚ,Bₚ2,B₀ = u
    d,G₀,λ,γ,Vₘₐₓ,Kₘ,ω,ω2,recovery = p
    du[1] =  -U(R,Vₘₐₓ,Kₘ)*B₀ - U(R, Vₘₐₓ*ω,Kₘ)*Bₚ -U(R,Vₘₐₓ*ω2,Kₘ)*Bₚ2
    du[2] = (1-λ)*G₀*ω*U(R,Vₘₐₓ*ω,Kₘ)*Bₚ + G₀*ω2*U(R,Vₘₐₓ*ω2,Kₘ)*Bₚ2  - d*Bₚ*(Bₚ+Bₚ2+B₀) + γ*B₀*Bₚ2 + recovery*Bₚ2
    du[3] =  γ*B₀*Bₚ -λ*G₀*ω2*U(R,Vₘₐₓ*ω2,Kₘ)*Bₚ2 - d*Bₚ2*(Bₚ+Bₚ2+B₀)  - recovery*Bₚ2 
    du[4] = G₀*U(R, Vₘₐₓ,Kₘ)*B₀ + λ*G₀*ω*U(R, Vₘₐₓ*ω,Kₘ)*Bₚ +λ*G₀*ω2*U(R,Vₘₐₓ*ω2,Kₘ)*Bₚ2 - γ*B₀*Bₚ - γ*B₀*Bₚ2 - d*B₀*(Bₚ+Bₚ2+B₀)
end

# Figure 1
plas_dens = zeros(50,50)
plas_dens2 = zeros(50,50)
plas_dens3 = zeros(50,50)
λ = 1.0e-8
for multiplier in [0.0]
    Threads.@threads for i in 1:50
        γ = range(1.0e-6,1.0e-4,50)[i]
        for (j,ω) in enumerate(range(0.95,1.0,50))
            p = [0.005,8e+8,λ,γ,6e-10,1.0,ω,ω, 0.0]
            prob = ODEProblem(simple_levin,[1.0,5.0,0.0,50.0],(0,100000),p)
            sol = solve(prob, Tsit5()).u[end]
            plas_dens[i,j] = sum(sol[2:3])

            p2 = [0.005,8e+8,λ,γ,6e-10,1.0,1.0,ω*multiplier, 0.0]
            prob2 = ODEProblem(simple_levin,[1.0,5.0,0.0,50.0],(0,100000),p2)
            sol2 = solve(prob2, Tsit5()).u[end]
            plas_dens2[i,j] = sum(sol2[2:3])

            p3 = [0.005,8e+8,λ,γ,6e-10,1.0,ω,ω*multiplier, 0.0]
            prob3 = ODEProblem(simple_levin,[1.0,5.0,0.0,50.0],(0,100000),p3)
            sol3 = solve(prob3, Tsit5()).u[end]
            plas_dens3[i,j] = sum(sol3[2:3])
        end
    end
end

fig = Figure()

ax = Axis(fig[1,1], title = "a")
ax.ylabel = "Conjugation rate γ"
ax.xlabel = "Relative fitness of plasmid host ω"
ax.yticks = (5:5:50,reverse(string.(round.(range(1.0e-6,1.0e-4,50)[5:5:50],sigdigits = 3))))
ax.xticks = (10:10:50,string.(round.(range(0.95,1.0,50)[10:10:50],sigdigits = 3)))
hm =heatmap!(ax,rotr90(plas_dens),colorrange = (0.0,48))

ax2 = Axis(fig[1,2], title = "b")
ax2.ylabel = "Conjugation rate γ"
ax2.xlabel = "Relative fitness of plasmid host ω = 1"
ax2.yticks = (5:5:50,reverse(string.(round.(range(1.0e-6,1.0e-4,50)[5:5:50],sigdigits = 3))))
heatmap!(ax2,rotr90(plas_dens2),colorrange = (0.0,48))
hidexdecorations!(ax2, label = false)

ax3 = Axis(fig[1,3], title = "c")
ax3.ylabel = "Conjugation rate γ"
ax3.xlabel = "Relative fitness of plasmid host ω"
ax3.yticks = (5:5:50,reverse(string.(round.(range(1.0e-6,1.0e-4,50)[5:5:50],sigdigits = 3))))
ax3.xticks = (10:10:50,string.(round.(range(0.95,1.0,50)[10:10:50],sigdigits = 3)))
heatmap!(ax3,rotr90(plas_dens3),colorrange = (0.0,48))

Colorbar(fig[1,4], hm,vertical = true, label = "Equilibrium plasmid density")

save("fig1.png",fig)

# Supplementary figures
t = time()
for multiplier in 0.0:0.1:0.9
    println([multiplier round(time()-t,sigdigits = 3)])
    for λ in range(1.0e-10,1.0e-5,5)
        for d in range(0.00005,0.05,5)
            for δ in range(0.000005,0.05,5)
            plas_dens = zeros(25,25)
            plas_dens2 = zeros(25,25)
            plas_dens3 = zeros(25,25)
            name = "ω_c = $multiplier, λ = $(round(λ,sigdigits=3)), d = $(round(d,sigdigits=3)), δ = $(round(δ,sigdigits=3))"
            Threads.@threads for i in 1:25
                γ = range(1.0e-6,1.0e-4,25)[i]
                for (j,ω) in enumerate(range(0.95,1.0,25))
                    p = [d,8e+8,λ,γ,6e-10,1.0,ω,ω, 0.0]
                    prob = ODEProblem(simple_levin,[1.0,5.0,0.0,50.0],(0,100000),p)
                    sol = solve(prob, Tsit5()).u[end]
                    plas_dens[i,j] = sum(sol[2:3])

                    p2 = [d,8e+8,λ,γ,6e-10,1.0,1.0,ω*multiplier, 0.0]
                    prob2 = ODEProblem(simple_levin,[1.0,5.0,0.0,50.0],(0,100000),p2)
                    sol2 = solve(prob2, Tsit5()).u[end]
                    plas_dens2[i,j] = sum(sol2[2:3])

                    p3 = [d,8e+8,λ,γ,6e-10,1.0,ω,ω*multiplier, 0.0]
                    prob3 = ODEProblem(simple_levin,[1.0,5.0,0.0,50.0],(0,100000),p3)
                    sol3 = solve(prob3, Tsit5()).u[end]
                    plas_dens3[i,j] = sum(sol3[2:3])
                end
            end
            fig = Figure(size = (1153,450), title = name)
            max_dens = maximum(maximum.([plas_dens,plas_dens2,plas_dens3]))

            ax = Axis(fig[1,1], title = "a")
            ax.ylabel = "Conjugation rate γ"
            ax.xlabel = "Relative fitness of plasmid ω"
            ax.yticks = (5:5:50,reverse(string.(round.(range(1.0e-6,1.0e-4,50)[5:5:50],sigdigits = 3))))
            ax.xticks = (10:10:50,string.(round.(range(0.95,1.0,50)[10:10:50],sigdigits = 3)))
            hm =heatmap!(ax,rotr90(plas_dens),colorrange = (0.0,max_dens))

            ax2 = Axis(fig[1,2], title = "b")
            ax2.ylabel = "Conjugation rate γ"
            ax2.xlabel = "Relative fitness of plasmid ω = 1"
            ax2.yticks = (5:5:50,reverse(string.(round.(range(1.0e-6,1.0e-4,50)[5:5:50],sigdigits = 3))))
            heatmap!(ax2,rotr90(plas_dens2),colorrange = (0.0,max_dens))
            hidexdecorations!(ax2, label = false)

            ax3 = Axis(fig[1,3], title = "c")
            ax3.ylabel = "Conjugation rate γ"
            ax3.xlabel = "Relative fitness of plasmid ω"
            ax3.yticks = (5:5:50,reverse(string.(round.(range(1.0e-6,1.0e-4,50)[5:5:50],sigdigits = 3))))
            ax3.xticks = (10:10:50,string.(round.(range(0.95,1.0,50)[10:10:50],sigdigits = 3)))
            heatmap!(ax3,rotr90(plas_dens3),colorrange = (0.0,max_dens))

            Colorbar(fig[1,4], hm,vertical = true, label = "Equilibrium plasmid density")
            fig[0, :] = Label(fig, name)
            save(name * ".png",fig)
        end
        end
    end
end

