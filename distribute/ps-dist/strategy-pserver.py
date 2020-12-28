import os
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers.experimental.preprocessing as kpl


# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

# ==================================
# cluster setup
# ==================================
TF_CONFIG = os.getenv('TF_CONFIG')
print("TF_CONFIG: ", TF_CONFIG)

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
cluster_spec_dict = cluster_resolver.cluster_spec().as_dict()
print("cluster info : ", cluster_spec_dict)

NUM_PS = len(cluster_spec_dict['ps'])
print("ps nums: ", NUM_PS)

# ==================================
# create worker/ps
# ==================================
if cluster_resolver.task_type in ('worker', 'ps'):
  print("current role: ", cluster_resolver.task_type)
  server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol="grpc")
  # Blocking the process that starts a server from exiting.
  server.join()


# ===================================
# create coordinator
# ===================================

# create variable_partitioner
variable_partitioner = (
    tf.distribute.experimental.partitioners.FixedShardsPartitioner(
        num_shards=NUM_PS))

# parameterServerStrategy
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

# create coordinator
if cluster_resolver.task_type == 'chief':
  print("current role is coordinator.")
  coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
    strategy=strategy)

feature_vocab = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong",
    "wonder_woman"
]
label_vocab = ["yes", "no"]

with strategy.scope():
  feature_lookup_layer = kpl.StringLookup(vocabulary=feature_vocab)

  label_lookup_layer = kpl.StringLookup(vocabulary=label_vocab,
                                        num_oov_indices=0,
                                        mask_token=None)

  raw_feature_input = keras.layers.Input(
      shape=(3,), dtype=tf.string, name="feature")
  feature_id_input = feature_lookup_layer(raw_feature_input)
  feature_preprocess_stage = keras.Model(
      {"features": raw_feature_input}, feature_id_input)

  raw_label_input = keras.layers.Input(
      shape=(1,), dtype=tf.string, name="label")
  label_id_input = label_lookup_layer(raw_label_input)
  label_preprocess_stage = keras.Model({"label": raw_label_input}, label_id_input)

def feature_and_label_gen(num_examples=200):
  examples = {"features": [], "label": []}
  for _ in range(num_examples):
    features = random.sample(feature_vocab, 3)
    label = ["yes"] if "avenger" in features else ["no"]
    examples["features"].append(features)
    examples["label"].append(label)
  return examples

examples = feature_and_label_gen() 

def dataset_fn(_):
  raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

  train_dataset = raw_dataset.map(
      lambda x: (
          {"features": feature_preprocess_stage(x["features"])},
          label_preprocess_stage(x["label"])
      )).shuffle(200).batch(32).repeat()
  return train_dataset

  # These variables created under the `strategy.scope` will be placed on parameter
# servers in a round-robin fashion.
with strategy.scope():
  # Create the model. The input needs to be compatible with KPLs.
  model_input = keras.layers.Input(
      shape=(3,), dtype=tf.int64, name="model_input")

  emb_layer = keras.layers.Embedding(
      input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=20)
  emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
  dense_output = keras.layers.Dense(units=1, activation="sigmoid")(emb_output)
  model = keras.Model({"features": model_input}, dense_output)

  optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
  accuracy = keras.metrics.Accuracy()

@tf.function
def step_fn(iterator):

  def replica_fn(iterator):
    batch_data, labels = next(iterator)
    with tf.GradientTape() as tape:
      pred = model(batch_data, training=True)
      per_example_loss = keras.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
      loss = tf.nn.compute_average_loss(per_example_loss)
      gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    accuracy.update_state(labels, actual_pred)
    return loss

  losses = strategy.run(replica_fn, args=(iterator,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)

num_epoches = 10000
steps_per_epoch = 5
for i in range(num_epoches):
  accuracy.reset_states()
  for _ in range(steps_per_epoch):
    coordinator.schedule(step_fn, args=(per_worker_iterator,))
  # Wait at epoch boundaries.
  coordinator.join()
  print ("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))

loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
print ("Final loss is %f" % loss.fetch())
