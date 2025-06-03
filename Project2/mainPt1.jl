include("compressionLib.jl")
using Plots
using FFTW
using Statistics
using JSON

const OUTPUT_DIR = "Project2/results/"
matrix_sizes = [10]
iterations = 10

function save_results(results::Dict, filename::String)
    open(filename, "w") do file
        write(file, JSON.json(results))
    end
end

function dct_time(size, iterations)
    custom_times = Vector{Float64}(undef, iterations)
    fftw_times = Vector{Float64}(undef, iterations)

    for i in 1:iterations
        data = rand(size, size)

        custom_times[i] = @elapsed begin DCT2(data) end
        fftw_times[i] = @elapsed begin FFTW.dct(data) end

        println("iterazione $i con grandezza $size")
    end

    return custom_times, fftw_times
end

for size in matrix_sizes
    custom, fft = dct_time(size, iterations)
    results = Dict(
        "custom_mean" => mean(custom),
        "fft_mean"    => mean(fft),
        "size"        => size
    )
        
    isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)
    filename = OUTPUT_DIR * string(size) * "_customDCT.json"
    save_results(results, filename)
end

custom_means = Float64[]
fft_means = Float64[]

for size in matrix_sizes
    filepath = OUTPUT_DIR * "$(size)_customDCT.json"
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

plot!(legend=:topleft)
savefig(OUTPUT_DIR * "dct2_plot.png")
println("Grafico salvato in 'results/dct2_plot.png'")

#TEST DCT1 e DCT2 functions

v = [231 32 233 161 24 71 140 245]
println(FFTW.dct(v))


A = [231 32 233 161 24 71 140 245
    247 40 248 245 124 204 36 107
    234 202 245 167 9 217 239 173
    193 190 100 167 43 180 8 70
    11 24 210 177 81 243 8 112
    97 195 203 47 125 114 165 181
    193 70 174 167 41 30 127 245
    87 149 57 192 65 129 178 228]
B = FFTW.dct(A)
println("\nDCT2\n", B)

for i = eachindex(B[:, 1])
     println(B[i, :])
end


