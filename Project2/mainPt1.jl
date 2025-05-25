include("compressionLib.jl")
using Plots
using FFTW
using Statistics
using JSON

matrix_sizes = [10, 20, 40, 80, 160, 320, 640]
iterations = 10

function dct(size, iterations)
    custom_times = Vector{Float64}(undef, iterations)
    fftw_times = Vector{Float64}(undef, iterations)

    for i in 1:iterations
        data = rand(size, size)

        custom_times[i] = @elapsed DCT2(data)
        fftw_times[i] = @elapsed dct(data)

        println("iterazione $i con grandezza $size")
    end

    return custom_times, fftw_times
end

for size in matrix_sizes
    custom, fft = dct(size, iterations)
        result_summary = Dict(
        "custom_mean" => mean(custom),
        "fft_mean"    => mean(fft),
        "size"        => size
    )

    output_file = joinpath("results", "$(size).json")
    open(output_file, "w") do file
        write(file, JSON.json(result_summary))
    end
end

custom_means = Float64[]
fft_means = Float64[]

for size in matrix_sizes
    filepath = "results/$(size).json"
    data = JSON.parsefile(filepath)

    push!(custom_means, data["custom_mean"])
    push!(fft_means, data["fft_mean"])
end

plot(
    matrix_sizes, custom_means;
    label = "Custom DCT2", lw = 2, marker = :circle,
    xlabel = "Matrix size (N)", ylabel = "Time (s) [log scale]",
    title = "Execution Time of DCT2 Algorithms",
    yscale = :log10
)

plot!(
    matrix_sizes, fft_means;
    label = "FFTW DCT2", lw = 2, marker = :diamond
)

savefig("results/dct2_plot.png")
println("Grafico salvato in 'results/dct2_plot.png'")
