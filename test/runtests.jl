using NLSBifurcations
using Test

@testset "NLSBifurcations" begin
    a, b = -1.0, 1.0

    @testset "Potentials" begin
        V = square_well(a, b, -2.0)
        @test V(0.0) == -2.0
        @test V(1.5) == 0.0

        Vp = paper_potential(3.0, 0.0)
        @test Vp(0.0) < 0  # well
        @test Vp(2.0) == 0.0  # outside support
    end

    @testset "Scattering data (square well)" begin
        V = square_well(a, b, -2.0)
        data = compute_all_scattering_data(a, b, V; N=300, k_max=3.0, verbose=false)

        @test length(data.bound_poles) == 1
        @test length(data.scat_poles) == 1
        @test data.bound_poles[1].κ > 0
        @test data.scat_poles[1].κ < 0
    end

    @testset "Scattering threshold transition" begin
        # Below threshold: 1 bound + 1 resonance
        V1 = square_well(a, b, -2.0)
        d1 = compute_all_scattering_data(a, b, V1; N=300, k_max=3.0, verbose=false)
        @test length(d1.bound_poles) == 1
        @test length(d1.scat_poles) == 1

        # Above threshold: 2 bound + 0 resonance
        V2 = square_well(a, b, -3.0)
        d2 = compute_all_scattering_data(a, b, V2; N=300, k_max=3.0, verbose=false)
        @test length(d2.bound_poles) == 2
        @test length(d2.scat_poles) == 0
    end

    @testset "Seed finding" begin
        V = square_well(a, b, -2.0)
        seeds = find_all_seeds(a, b, V; E_list=[-0.5], ζmax=8.0, nscan=2000, N=3000, tolH=1e-6)
        @test length(seeds) >= 1
    end
end
