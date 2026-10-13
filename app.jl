module ConcSynthApp

using GenieFramework
using JSON3
using DataFrames
using CSV

@genietools

include(joinpath(@__DIR__, "src", "ConcSynthCore.jl"))
using .ConcSynthCore

const DEFAULT_TRAJECTORIES = Dict(
    "density" => [(0.0, 1.0), (1.0, 1.0)],
    "k_nearest" => [(0.0, 0.0), (1.0, 0.0)],
    "weight_pitch_hz" => [(0.0, 1.0), (1.0, 1.0)],
    "weight_loudness_db" => [(0.0, 1.0), (1.0, 1.0)],
    "weight_harmonicity" => [(0.0, 1.0), (1.0, 1.0)],
    "weight_spectral_centroid_hz" => [(0.0, 1.0), (1.0, 1.0)],
)

const DEFAULT_TRAJECTORIES_JSON = JSON3.write(DEFAULT_TRAJECTORIES)

@app begin
    @in corpus_audio_path = "./audio/corpus.wav"
    @in corpus_csv_path = "./data/corpus_descriptors.csv"
    @in target_path = "./audio/target.wav"
    @in output_wav = "./output/resynth.wav"
    @in target_analysis_csv_path = "./data/target_analysis.csv"

    @in descriptor_names_text = join(default_descriptor_names(), ",")
    @in trajectories_json = DEFAULT_TRAJECTORIES_JSON

    @out status_text = "Ready. Load descriptor names or start sketching trajectories."
    @out result_json = "{}"

    @in load_descriptors_btn = false
    @in run_pipeline_btn = false

    @onbutton load_descriptors_btn begin
        names = descriptor_names_from_csv(corpus_csv_path)
        descriptor_names_text = join(names, ",")
        status_text = "Loaded $(length(names)) descriptor names."
    end

    @onbutton run_pipeline_btn begin
        result = run_pipeline_skeleton(
            corpus_audio_path = corpus_audio_path,
            corpus_csv_path = corpus_csv_path,
            target_path = target_path,
            output_wav = output_wav,
            target_analysis_csv_path = target_analysis_csv_path,
            trajectories_json = trajectories_json,
        )
        result_json = JSON3.write(result)
        status_text = result["message"]
    end
end

function input_card()
    card(style = "margin-bottom: 18px;", [
        card_section([h4("Paths and files"), p("These fields mirror your previous Flask tool.")]),
        card_section([
            textfield("Corpus audio path", :corpus_audio_path, id = "corpus_audio_path_input", filled = true),
            textfield("Corpus CSV path", :corpus_csv_path, id = "corpus_csv_path_input", filled = true),
            textfield("Target audio path", :target_path, id = "target_path_input", filled = true),
            textfield("Output WAV path", :output_wav, id = "output_wav_input", filled = true),
            textfield("Target analysis CSV path", :target_analysis_csv_path, id = "target_analysis_csv_path_input", filled = true),
        ])
    ])
end

function control_card()
    card(style = "margin-bottom: 18px;", [
        card_section([h4("Actions"), p("This skeleton only loads descriptor names and captures trajectories.")]),
        card_section([
            btn("Load descriptors", @click(:load_descriptors_btn), color = "primary"),
            btn("Run skeleton", @click(:run_pipeline_btn), color = "secondary", style = "margin-left: 12px;"),
            p("{{status_text}}", style = "margin-top: 12px; font-weight: 600;"),
        ])
    ])
end

function hidden_state_card()
    card(style = "margin-bottom: 18px;", [
        card_section([h4("Reactive state"), p("The canvas script reads and writes these fields.")]),
        card_section([
            textfield("Descriptor names (comma separated)", :descriptor_names_text, id = "descriptor_names_text", filled = true),
            textarea("Trajectory JSON", :trajectories_json, id = "trajectories_json", autogrow = true, filled = true),
            textarea("Latest result JSON", :result_json, id = "result_json", autogrow = true, filled = true),
        ])
    ])
end

function trajectory_shell()
    card(style = "margin-bottom: 18px;", [
        card_section([h4("Trajectories"), p("Draw curves in the canvas below. The browser script stores them in trajectories_json.")]),
        card_section([
            Genie.Renderer.Html.div("", id = "trajectory-app", class = "trajectory-app"),
            Genie.Renderer.Html.script(src = "/js/trajectory_editor.js"),
        ])
    ])
end

function ui()
    page(
        title = "ConcSynth Genie/Stipple Skeleton",
        class = "container q-pa-md",
        [
            Genie.Renderer.Html.link(rel = "stylesheet", href = "/css/app.css"),
            h2("Concatenative Synthesis Skeleton"),
            p("A Genie/Stipple starter that mirrors your previous HTML layout and keeps room for the full DSP pipeline."),
            input_card(),
            control_card(),
            trajectory_shell(),
            hidden_state_card(),
        ],
    )
end

@page("/", ui)

end

using .ConcSynthApp
Genie.isrunning() || up()
