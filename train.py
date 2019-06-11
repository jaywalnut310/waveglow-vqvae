import tensorflow as tf

from hparams import create_hparams
import data
import utils
import waveglow as model


def main(argv):
  args = utils.parse_args("Train a wavenet model")
  utils.redirect_log_to_file(args.model_dir)

  hparams = create_hparams(args.model_dir, args.configs, initialize=True)
  utils.check_git_hash(args.model_dir)

  # Prepare data
  data.load_vocab(hparams)
  train_input_fn = data.InputPipeline(hparams, tf.estimator.ModeKeys.TRAIN)
  eval_input_fn = data.InputPipeline(hparams, tf.estimator.ModeKeys.EVAL)

  # Training
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
      max_steps=hparams.train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
      steps=hparams.eval_steps,
      throttle_secs=hparams.throttle_secs)

  distribution = tf.contrib.distribute.MirroredStrategy()
  run_config = tf.estimator.RunConfig(model_dir=args.model_dir,
      train_distribute=distribution,
      save_summary_steps=hparams.save_summary_steps,
      save_checkpoints_secs=hparams.save_checkpoints_secs,
      keep_checkpoint_max=hparams.n_checkpoints)
  estimator = tf.estimator.Estimator(
      model_fn=model.build_model_fn(hparams),
      config=run_config,
      model_dir=args.model_dir)

  tf.estimator.train_and_evaluate(
      estimator, 
      train_spec, 
      eval_spec)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
