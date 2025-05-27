include("compressionLib.jl")

using Dash
using FFTW, Images, Base64, ImageView
using JSON

function create_layout()
    return html_div(
        style=Dict("text-align" => "center", "font-family" => "Verdana"),
        children=[
            html_h2("Seleziona parametri e carica un'immagine BMP"),
            html_div([
                html_label("Blocco F:"),
                dcc_input(id="input-block-size", type="number", style=Dict("margin" => "10px"))
            ]),
            html_div([
                html_label("Soglia di taglio d:"),
                dcc_input(id="input-cutoff", type="number", style=Dict("margin" => "10px"))
            ]),
            html_div([
                html_label("Caricamento immagine:"),
                dcc_upload(
                    id="image-upload",
                    children=html_button("Sfoglia", style=Dict("margin-left" => "10px"))
                )
            ]),
            html_button("Avvia Compressione", id="btn-process", style=Dict("margin-top" => "20px")),
            html_div(id="status-msg", style=Dict("margin-top" => "25px", "color" => "blue"))
        ]
    )
end

app = dash()
app.layout = create_layout()

callback!(
    app,
    Output("status-msg", "children"),
    Input("btn-process", "n_clicks"),
    State("image-upload", "contents"),
    State("input-block-size", "value"),
    State("input-cutoff", "value")
) do click_count, encoded_img, block_size, cutoff
    if isnothing(click_count) || isnothing(encoded_img) || isnothing(block_size) || isnothing(cutoff)
        return "Assicurati di aver caricato l'immagine e inserito entrambi i parametri."
    end

    if cutoff < 0 || cutoff > (2 * block_size - 2)
        return "Parametro d fuori intervallo valido (0 ≤ d ≤ 2F−2)"
    end

    try
        # Decodifica immagine da base64
        content_parts = split(encoded_img, ",")
        img_bytes = Base64.base64decode(content_parts[end])
        path_temp = "temp_input.bmp"

        open(path_temp, "w") do file
            write(file, img_bytes)
        end

        # Caricamento immagine
        img = load(path_temp)
        gray_img = Gray.(img)
        imshow(gray_img, name="Originale")

        # Compressione
        fragments = image_to_blocks(gray_img, block_size)
        compressed = image_compress(fragments, cutoff)

        img_h, img_w = size(gray_img)
        valid_h = block_size * fld(img_h, block_size)
        valid_w = block_size * fld(img_w, block_size)

        output_img = reassemble_image(compressed, valid_h, valid_w, block_size)
        output_img = Gray{N0f8}.(output_img)
        imshow(output_img, name="Compressa")

        return "Compressione completata e immagini visualizzate."

    catch err
        @warn "Errore nel processo: $err"
        return "Errore nella gestione dell'immagine. Riprova."
    end
end

run_server(app, "0.0.0.0", debug=true)
