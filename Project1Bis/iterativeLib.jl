using SparseArrays
using LinearAlgebra

function load_matrix(file_path)
    open(file_path) do file
        header = readline(file)
        rows, cols, values = split(header, "  ")
        rows = parse(Int32, rows)
        cols = parse(Int32, cols)
        values = parse(Int32, values)
        
        matrix = zeros(rows, cols)
        for line in eachline(file)
            i, j, v = split(line, "  ")
            i = parse(Int32, i)
            j = parse(Int32, j)
            v = parse(Float64, v)
            matrix[i, j] = v
        end
        return matrix
    end
end

function load_sparse_matrix(file_path)
    open(file_path) do file
        header = readline(file)
        rows, cols, values = split(header, "  ")
        rows = parse(Int32, rows)
        cols = parse(Int32, cols)
        values = parse(Int32, values)
        
        matrix = spzeros(rows, cols)
        for line in eachline(file)
            i, j, v = split(line, "  ")
            i = parse(Int32, i)
            j = parse(Int32, j)
            v = parse(Float64, v)
            matrix[i, j] = v
        end
        return matrix
    end
end

function jacobi_method(A, b, tolerance, max_iterations=20000)
    println("Executing Jacobi Method")

    x = zeros(size(A, 1))
    inv_diag = spdiagm(0 => 1.0 ./ diag(A))
    iterations = 0

    for k = 1:max_iterations
        residual = b - A * x
        x = x + inv_diag * residual

        if norm(residual) / norm(b) < tolerance
            iterations = k
            break
        end
    end

    return x, iterations
end

function gauss_seidel_method(A, b, tolerance, max_iterations=20000)
    println("Executing Gauss-Seidel Method")

    x = zeros(size(A, 1))
    lower_triangle = tril(A)
    residual = lower_triangle - A
    iterations = 0

    for k = 1:max_iterations
        residual = b - A * x
        correction = forward_substitution(lower_triangle, residual)
        x = x + correction

        if norm(residual) / norm(b) < tolerance
            iterations = k
            break
        end
    end

    return x, iterations
end

function gradient_descent_method(A, b, tolerance, max_iterations=20000)
    println("Executing Gradient Descent")

    x = zeros(size(A, 1))
    iterations = 0

    for k = 1:max_iterations
        residual = b - A * x
        step_size = dot(residual, residual) / dot(residual, A * residual)
        x = x + step_size * residual

        if norm(residual) / norm(b) < tolerance
            iterations = k
            break
        end
    end

    return x, iterations
end

function conjugate_gradient_method(A, b, tolerance, max_iterations=20000)
    println("Executing Conjugate Gradient Method")

    x = zeros(size(A, 1))
    direction = b - A * x
    iterations = 0

    for k = 1:max_iterations
        residual = b - A * x
        step_size = dot(direction, residual) / dot(direction, A * direction)
        x = x + step_size * direction

        new_residual = b - A * x
        beta = dot(direction, A * new_residual) / dot(direction, A * direction)
        direction = new_residual - beta * direction

        if norm(residual) / norm(b) < tolerance
            iterations = k
            break
        end
    end

    return x, iterations
end

function get_pivot_index(U, pivot_type, k)
    n = size(U, 1)
    pivot = k
    if pivot_type == "partial"
        max_value = abs(U[k, k])
        for i = k+1:n
            if abs(U[i, k]) > max_value
                max_value = abs(U[i, k])
                pivot = i
            end
        end
    end
    if pivot_type == "total"
        max_value = abs(U[k, k])
        for i = k+1:n
            for j = k+1:n
                if abs(U[i, j]) > max_value
                    max_value = abs(U[i, j])
                    pivot = i
                end
            end
        end
    end
    return pivot
end

function generate_b_vector(A)
    x = ones(size(A, 1))
    b = A * x
    return b
end

function forward_substitution(A, b)
    n = size(A, 1)
    x = zeros(n)
    x[1] = b[1] / A[1, 1]
    for i = 2:n
        x[i] = b[i]
        for j = 1:i-1
            x[i] -= A[i, j] * x[j]
        end
        x[i] /= A[i, i]
    end
    return x
end
