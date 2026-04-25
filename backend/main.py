import os
os.environ.setdefault("MKL_THREADING_LAYER", "TBB")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
from scripts.pipeline.train import run_train
from scripts.pipeline.evaluate import run_evaluate
from core.config.settings import config

def main():
    parser = argparse.ArgumentParser(description="v5 pipeline orchestrator")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--use_huggingface', action='store_true', default=config.use_huggingface)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--max_length', type=int, default=config.max_sequence_length)
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)
    parser.add_argument('--use_class_weights', action='store_true', default=config.use_class_weights)
    parser.add_argument('--local_csv', type=str, default=None)
    parser.add_argument('--warmup_ratio', type=float, default=config.warmup_ratio)
    parser.add_argument('--dropout', type=float, default=config.dropout)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--use_weighted_sampler', action='store_true', default=config.use_weighted_sampler)
    parser.add_argument('--eval_interval', type=int, default=None)
    parser.add_argument('--eval_max_batches', type=int, default=None)
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        args.train = True
        args.evaluate = True
        
    if args.train:
        run_train(use_huggingface=args.use_huggingface, sample_size=args.sample_size, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, learning_rate=args.learning_rate, weight_decay=args.weight_decay, use_class_weights=args.use_class_weights, file_path=args.local_csv, warmup_ratio=args.warmup_ratio, dropout=args.dropout, gradient_accumulation_steps=args.grad_accum, use_weighted_sampler=args.use_weighted_sampler, eval_interval=args.eval_interval, eval_max_batches=args.eval_max_batches)
    if args.evaluate:
        run_evaluate(use_huggingface=args.use_huggingface, sample_size=args.sample_size, batch_size=args.batch_size, max_length=args.max_length)

if __name__ == '__main__':
    main()
