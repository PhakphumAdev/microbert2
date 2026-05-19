// --------------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------------
local language = "maltese";
local experiment_name = "tlm";

// Rclone Upload Configuration ---------------------------------------------------
local rclone_remote_path = null;

// Tokenization -------------------------------------------------------------------
local stanza_retokenize = false;
local stanza_use_mwt = false;
local stanza_language_code = null;
// Shared bilingual vocab: 2x the monolingual vocab size (10000) to cover both languages.
local vocab_size = 20000;

// Data ---------------------------------------------------------------------------
local whitespace_tokenized_text_path_train = "../slate/mlt/train.txt";
local whitespace_tokenized_text_path_dev = "../slate/mlt/dev.txt";
local train_conllu_path = "../slate/mlt/mlt_mudt-ud-train.conllu";
local dev_conllu_path = "../slate/mlt/mlt_mudt-ud-dev.conllu";
local test_conllu_path = "../slate/mlt/mlt_mudt-ud-test.conllu";
// Parallel data reuses the same tsv files as the MT task
local train_mt_path = "../slate/mlt/train.tsv";
local dev_mt_path = "../slate/mlt/dev.tsv";
local test_mt_path = "../slate/mlt/test.tsv";

// Encoder ------------------------------------------------------------------------
local max_length = 512;
local hidden_size = 128;
local num_layers = 4;
local bert_type = "modernbert";
local bert_config = {
    hidden_size: hidden_size,
    num_hidden_layers: num_layers,
    num_attention_heads: 8,
    intermediate_size: hidden_size + hidden_size / 2,
    max_position_embeddings: max_length,
    attention_dropout: 0.1,
    embedding_dropout: 0.1,
    mlp_dropout: 0.1,
    global_attn_every_n_layers: 1,
};

// Training and Optimization ------------------------------------------------------
local batch_size = 128;
local grad_accum = 1;
local effective_batch_size = grad_accum * batch_size;
local num_steps = 150000;
local validate_every = 5000;

local optimizer = {
    type: "torch::AdamW",
    lr: 3e-3,
    betas: [0.9, 0.98],
    eps: 1e-6,
    weight_decay: 0.05
};
local lr_scheduler = {
    type: "transformers::reduce_lr_on_plateau",
    factor: 0.8,
    patience: 5,
    min_lr: 1e-5,
};
local loss_auto_scaling = false;

// Some set up, don't modify ------------------------------------------------------
local util = import 'lib/util.libsonnet';
local model_path = (
    "./workspace/models/" + language + "_" + experiment_name + "_" + util.stringifyObject(bert_config)
);
local tokenizer = { pretrained_model_name_or_path: model_path };

// Tasks --------------------------------------------------------------------------
local mlm_task = {
    type: "microbert2.microbert.tasks.mlm.MLMTask",
    dataset: { type: "ref", ref: "raw_text_data" },
    tokenizer: tokenizer,
};
local pos_task = {
    type: "microbert2.microbert.tasks.ud_pos.UDPOSTask",
    head: {
        num_layers: num_layers,
        embedding_dim: hidden_size,
        use_layer_mix: false,
        layer_index: 1,
    },
    tag_type: "xpos",
    train_conllu_path: train_conllu_path,
    dev_conllu_path: dev_conllu_path,
    test_conllu_path: test_conllu_path,
    proportion: 0.2,
};
local parser = (import "lib/parser.libsonnet")(hidden_size, num_layers);
local parse_task = {
    type: "microbert2.microbert.tasks.ud_parse.UDParseTask",
    head: parser,
    train_conllu_path: train_conllu_path,
    dev_conllu_path: dev_conllu_path,
    test_conllu_path: test_conllu_path,
};
// TLM task: reads the same parallel tsv as the MT tasks but uses masked LM over
// the concatenated [CLS] src [SEP] tgt [SEP] sequence instead of a decoder.
local tlm_task = {
    type: "microbert2.microbert.tasks.tlm.TLMTask",
    dataset: { type: "ref", ref: "parallel_text_data" },
    tokenizer: tokenizer,
    proportion: 0.2,
};
local tasks = [mlm_task, tlm_task];


// --------------------------------------------------------------------------------
// Internal--don't modify below here unless you're sure you know what you're doing!
// --------------------------------------------------------------------------------
local model = {
    type: "microbert2.microbert.model.model::microbert_model",
    tokenizer: tokenizer,
    model_output_path: model_path,
    loss_auto_scaling: loss_auto_scaling,
    tasks: tasks,
    encoder: {
        type: bert_type,
        tokenizer: tokenizer,
        bert_config: bert_config,
    }
};

local training_engine = {
    type: "mb2",
    optimizer: optimizer,
    lr_scheduler: lr_scheduler,
    amp: false,
    max_grad_norm: 10,
};

local collate_fn = {
    type: "microbert2.data.collator::collator",
    tokenizer: tokenizer,
    tasks: tasks,
};
local train_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
};
local val_dataloader = {
    shuffle: false,
    batch_size: batch_size,
    collate_fn: collate_fn,
    pin_memory: true,
};

{
    steps: {
        raw_text_data: {
            type: "microbert2.data.text::read_whitespace_tokenized_text",
            train_path: whitespace_tokenized_text_path_train,
            dev_path: whitespace_tokenized_text_path_dev,
            test_path: whitespace_tokenized_text_path_dev,
            stanza_retokenize: stanza_retokenize,
            stanza_use_mwt: stanza_use_mwt,
            stanza_language_code: stanza_language_code,
        },
        // Parallel data for TLM -- reuses the same tsv files as the MT task
        parallel_text_data: {
            type: "microbert2.data.text::read_parallel_tsv",
            train_path: train_mt_path,
            dev_path: dev_mt_path,
            test_path: test_mt_path,
        },
        tokenizer: {
            type: "microbert2.data.tokenize::train_tokenizer",
            dataset: { "type": "ref", "ref": "raw_text_data" },
            vocab_size: vocab_size,
            model_path: model_path,
            // Passing tlm_task here ensures English target tokens are included
            // in the shared bilingual vocabulary.
            tasks: tasks,
        },
        tokenized_text_data: {
            type: "microbert2.data.tokenize::subword_tokenize",
            dataset: { "type": "ref", "ref": "raw_text_data" },
            max_length: max_length,
            tokenizer: { "type": "ref", "ref": "tokenizer" },
            tasks: tasks,
        },

        // Merge inputs
        model_inputs: {
            type: "microbert2.data.combine::combine_datasets",
            datasets: { "type": "ref", "ref": "tokenized_text_data" },
            tasks: tasks,
        },

        // Begin training
        trained_model: {
            type: "microbert2.train::train",
            model: model,
            dataset_dict: { type: "ref", ref: "model_inputs" },
            training_engine: training_engine,
            log_every: 1,
            train_dataloader: train_dataloader,
            train_steps: num_steps,
            grad_accum: grad_accum,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            validation_split: "dev",
            validation_dataloader: val_dataloader,
            val_metric_name: "mlm_perplexity",
            auto_aggregate_val_metric: false,
            callbacks: [
                {
                    type: "microbert2.microbert.model.model::write_model",
                    path: model_path,
                    model_attr: "encoder.encoder"
                },
                {type: "microbert2.microbert.model.model::reset_metrics"},
                {
                    type: "microbert2.microbert.model.model::rclone_upload",
                    remote_path: rclone_remote_path,
                    upload_logs: true
                },
                {
                    type: "microbert2.microbert.model.model::huggingface_upload",
                    repo_id: "pakphum/microbert-maltese-tlm",
                    private: false,
                    commit_message: "Upload microbert maltese TLM"
                },
            ],
        },
    }
}
