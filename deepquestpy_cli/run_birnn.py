import argparse
import os
from allennlp.common.util import import_module_and_submodules
import_module_and_submodules('deepquestpy')

from allennlp.commands.train import train_model_from_file
from allennlp.training.util import evaluate
from allennlp.models.archival import load_archive
from allennlp.data.data_loaders import SimpleDataLoader

def main(args):
    # Training
    if args.do_train:
        train_model_from_file(parameter_filename=args.config_file,
                              serialization_dir = args.output_dir,
                              force=args.overwrite_output_dir)
    # Evaluation
    if args.do_eval:
        archive = load_archive(args.eval_model)
        model = archive.model
        reader = archive.validation_dataset_reader if "validation_dataset_reader" in archive else archive.dataset_reader
        eval_instances = list(reader.read(args.eval_data_path))
        batch_size = archive.config["data_loader"]["batch_sampler"]["batch_size"]
        eval_loader = SimpleDataLoader(eval_instances, batch_size=batch_size)
        eval_loader.index_with(model.vocab)
        metrics = evaluate(model,eval_loader, predictions_output_file=args.pred_output_file)
        if (args.eval_output_file):
            with open(args.eval_output_file,mode="w", encoding="utf-8") as eval_results_fh:
                print(metrics,file=eval_results_fh)
                print("Evaluation results written to ", args.eval_output_file)
        else:
            print(metrics)
        
        if (args.pred_output_file):
            print ("Predictions are written to :", args.pred_output_file)
    return

def cli_main(args):
    if not args.do_train and not args.do_eval:
        raise ValueError("Please specify either --do_train to train or --do_eval to evaluate")

    if ( os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--config_file", type=str, default="deepquestpy/config/birnn_word.jsonnet", help="Experiment config file.")
    parser.add_argument("--output_dir", type=str,default="data/output/model", help="Directory to save trained models.")
    parser.add_argument("--overwrite_output_dir", action="store_true")

    # evaluation arguments
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_model", type=str,default="data/output/model/model.tar.gz", help="Model to evaluate.")
    parser.add_argument("--eval_output_file", type=str,default="data/output/eval_results.json", help="Output file to which evaluation results will be written.")

    parser.add_argument("--pred_output_file", type=str,default="data/output/predictions.txt", help="Output file to which test predictions are written")

    parser.add_argument("--eval_data_path", type=str,default="test", help="Path containing evaluation data (relative to data_path as set in the config_file")
    args = parser.parse_args()
    cli_main(args)
