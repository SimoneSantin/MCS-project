function DCT1(v)
    N = length(v)
    result = zeros(N)

    sqrt_1_N = sqrt(1 / N)
    sqrt_2_N = sqrt(2 / N)

    for k in 1:N
        acc = 0.0
        for i in 1:N
            acc += v[i] * cos(pi * (2*(i - 1) + 1) * (k - 1) / (2 * N))
        end
        result[k] = acc * (k == 1 ? sqrt_1_N : sqrt_2_N)
    end

    return result
end

function DCT2(A)
    N, M = size(A)
    transformed = zeros(N, M)

    for n in 1:N
        transformed[n, :] = DCT1(A[n, :])
    end

    for m in 1:M
        transformed[:, m] = DCT1(transformed[:, m])
    end

    return transformed
end

function image_to_blocks(img, f)
    h, w = size(img)

    h = f * fld(h, f)
    w = f * fld(w, f)

    blocks = []

    for x in 0:f:w - f
        for y in 0:f:h - f
            push!(blocks, @view img[y+1:y+f, x+1:x+f])
        end
    end

    return blocks
end

function image_compress(blocks, d)
    result = []

    for block in blocks
        b = Float64.(Array(block))
        c = FFTW.dct(b)

        for i in 1:size(c, 1), j in 1:size(c, 2)
            if i + j >= d
                c[i, j] = 0
            end
        end

        recovered = FFTW.idct(c)
        clamped = clamp.(recovered, 0.0, 1.0)
        push!(result, clamped)
    end

    return result
end

function reassemble_image(blocks, height, width, block_size)
    output = zeros(Gray, Int(height), Int(width))
    i = 1

    for col in 1:block_size:width
        for row in 1:block_size:height
            b = blocks[i]
            bh, bw = size(b)
            output[row:row+bh-1, col:col+bw-1] .= b
            i += 1
        end
    end

    return output
end
