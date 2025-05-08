include("iterativeLib.jl")

using DelimitedFiles
using Printf
using Statistics
using JSON
using StatsPlots
using Plots

const DATA_DIR = "Project1Bis/data/"
const OUTPUT_DIR = "Project1Bis/results/"
const MATRICES = ["spa1", "spa2", "vem1", "vem2"]
const TOLERANCES = [1e-4, 1e-6, 1e-8, 1e-10]

function relative_error(x_approx, x_exact)
    return norm(x_approx - x_exact) / norm(x_exact)
end

function save_results(results::Dict, filename::String)
    open(filename, "w") do file
        write(file, JSON.json(results))
    end
end

for matrix_name in MATRICES
    matrix_path = DATA_DIR * matrix_name * ".mtx"
    A = load_sparse_matrix(matrix_path)  
    b = generate_b_vector(A)  
    x_exact = ones(length(b))

    for tol in TOLERANCES
        tol_label = @sprintf("%.0e", tol)
        println("Solving matrix $matrix_name with tolerance $tol_label")

        jacobi_time = @elapsed x_jacobi, k_jacobi = jacobi_method(A, b, tol)  
        gs_time = @elapsed x_gs, k_gs = gauss_seidel_method(A, b, tol)  
        grad_time = @elapsed x_grad, k_grad = gradient_descent_method(A, b, tol)  
        cgrad_time = @elapsed x_cgrad, k_cgrad = conjugate_gradient_method(A, b, tol)  

        results = Dict(
            "jacobi_result" => x_jacobi,
            "jacobi_time" => jacobi_time,
            "jacobi_iterations" => k_jacobi,
            "jacobi_error" => relative_error(x_jacobi, x_exact),

            "gauss_seidel_result" => x_gs,
            "gauss_seidel_time" => gs_time,
            "gauss_seidel_iterations" => k_gs,
            "gauss_seidel_error" => relative_error(x_gs, x_exact),

            "gradient_result" => x_grad,
            "gradient_time" => grad_time,
            "gradient_iterations" => k_grad,
            "gradient_error" => relative_error(x_grad, x_exact),

            "conjugate_gradient_result" => x_cgrad,
            "conjugate_gradient_time" => cgrad_time,
            "conjugate_gradient_iterations" => k_cgrad,
            "conjugate_gradient_error" => relative_error(x_cgrad, x_exact)
        )

        isdir(OUTPUT_DIR) || mkpath(OUTPUT_DIR)
        filename = OUTPUT_DIR * matrix_name * "_" * tol_label * "_results.json"
        save_results(results, filename)
    end
end


for matrix_name in MATRICES
    errors_jacobi = Float64[]
    errors_gs = Float64[]
    errors_grad = Float64[]
    errors_cgrad = Float64[]

    for tol in TOLERANCES
        tol_label = @sprintf("%.0e", tol)
        filename = OUTPUT_DIR * matrix_name * "_" * tol_label * "_results.json"
        results = JSON.parsefile(filename)

        push!(errors_jacobi, results["jacobi_error"])
        push!(errors_gs, results["gauss_seidel_error"])
        push!(errors_grad, results["gradient_error"])
        push!(errors_cgrad, results["conjugate_gradient_error"])
    end

    plt = plot(
        TOLERANCES, errors_jacobi;
        label = "Jacobi", linewidth = 2, xscale = :log10, yscale = :log10,
        xticks = (TOLERANCES, string.(TOLERANCES)),
        yticks = 8
    )
    plot!(TOLERANCES, errors_gs; label = "Gauss-Seidel", linewidth = 2)
    plot!(TOLERANCES, errors_grad; label = "Gradient",  linewidth = 2)
    plot!(TOLERANCES, errors_cgrad; label = "Conjugate Gradient", linewidth = 2)

    title!("Error vs Tolerance - $(uppercase(matrix_name))")
    xlabel!("Tolerance")
    ylabel!("Relative Error")
    plot!(legend=:topleft, legendcolumns=1)
    savefig(OUTPUT_DIR * "$(matrix_name)_error_plot.png")
    println("Saved plot for matrix $matrix_name")
end

for matrix_name in MATRICES
    iterations_jacobi = Int[]
    iterations_gs = Int[]
    iterations_grad = Int[]
    iterations_cgrad = Int[]

    for tol in TOLERANCES
        tol_label = @sprintf("%.0e", tol)
        filename = OUTPUT_DIR * matrix_name * "_" * tol_label * "_results.json"
        results = JSON.parsefile(filename)

        push!(iterations_jacobi, results["jacobi_iterations"])
        push!(iterations_gs, results["gauss_seidel_iterations"])
        push!(iterations_grad, results["gradient_iterations"])
        push!(iterations_cgrad, results["conjugate_gradient_iterations"])
    end

    y_values = [iterations_jacobi iterations_gs iterations_grad iterations_cgrad]
    x_labels = string.(TOLERANCES)
    method_labels = repeat(["Jacobi", "Gauss-Seidel", "Gradient", "Conjugate Gradient"])

    plt = groupedbar(
        x_labels,
        y_values;
        bar_position = :dodge,
        label = method_labels,
        xlabel = "Tolerance",
        ylabel = "Number of Iterations",
        yscale = :log10,
        legend = :topleft,
        title = "Number of Iterations vs Tolerance - $(uppercase(matrix_name))",
        size = (900, 600)
    )

    n_methods = length(method_labels)
    x_vals = repeat(1:length(TOLERANCES), inner=n_methods)
    offsets = range(-0.8, stop=-0.2, length=n_methods)

    for (i, offset) in enumerate(offsets)
        for (j, y) in enumerate(y_values[:, i])
            x_pos = j + offset
            annotate!(x_pos, y * 1.3, text(string(y), :black, 8, :center))
        end
    end
    savefig(OUTPUT_DIR * "$(matrix_name)_iterations_barplot.png")
    println("Saved barplot with iterations for matrix $matrix_name")
end

for matrix_name in MATRICES
    iterations_jacobi = Float64[]
    iterations_gs = Float64[]
    iterations_grad = Float64[]
    iterations_cgrad = Float64[]

    for tol in TOLERANCES
        tol_label = @sprintf("%.0e", tol)
        filename = OUTPUT_DIR * matrix_name * "_" * tol_label * "_results.json"
        results = JSON.parsefile(filename)

        push!(iterations_jacobi, 1000 * results["jacobi_time"])
        push!(iterations_gs, 1000 * results["gauss_seidel_time"])
        push!(iterations_grad, 1000 * results["gradient_time"])
        push!(iterations_cgrad, 1000 * results["conjugate_gradient_time"])
    end

    y_values = [iterations_jacobi iterations_gs iterations_grad iterations_cgrad]
    x_labels = string.(TOLERANCES)
    method_labels = repeat(["Jacobi", "Gauss-Seidel", "Gradient", "Conjugate Gradient"])

    plt = groupedbar(
        x_labels,
        y_values;
        bar_position = :dodge,
        label = method_labels,
        xlabel = "Tolerance",
        ylabel = "Time required (ms)",
        yscale = :log10,
        legend = :topleft,
        title = "Time required (ms) vs Tolerance - $(uppercase(matrix_name))",
        size = (900, 600)
    )

    n_methods = length(method_labels)
    x_vals = repeat(1:length(TOLERANCES), inner=n_methods)
    offsets = range(-0.8, stop=-0.2, length=n_methods)

    for (i, offset) in enumerate(offsets)
        for (j, y) in enumerate(y_values[:, i])
            x_pos = j + offset
            annotate!(x_pos, y * 1.3, text(@sprintf("%.1f", y), :black, 8, :center))
        end
    end
    savefig(OUTPUT_DIR * "$(matrix_name)_time_barplot.png")
    println("Saved barplot with time for matrix $matrix_name")
end

