import tensorflo as tf
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_image(fname, label):
	img = tf.io.read_file(fname)
	img = tf.image.decode_jpeg(img, channels=3)
	img = (tf.cast(img, tf.float32)/127.5) - 1
	img = tf.image.resize(img, (128, 128))
	return img, label

def get_fnames_and_labels(data_folder):
	fnames = [x for x in data_folder.glob('**/*.jpg')]
	labels = [str(x.parent.name) for x in fnames]
	fnames = [str(x) for x in fnames]
	unique_labels = set(labels)
	indexes_to_names = dict((i, name) for name, i in zip(unique_labels, range(len(unique_labels))))
	names_to_indexes = dict((name, i) for i, name in indexes_to_names.items())
	labels = [names_to_indexes[x] for x in labels]
	return fnames, labels, indexes_to_names, names_to_indexes

def prepare_datasets(data_folder, batch_size):
	fnames, labels, indexes_to_names, names_to_indexes = get_fnames_and_labels(data_folder)
	train_fnames, val_fnames, train_labels, val_labels = train_test_split(fnames,
                                                                            labels,
                                                                            train_size=0.9,
                                                                            random_state=128)
	train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_fnames), tf.constant(train_labels)))
	val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_fnames), tf.constant(val_labels)))
	train_data = (train_data.map(load_and_preprocess_image)
             .shuffle(buffer_size=10000)
             .batch(batch_size)
	     .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	     .repeat()
             )
	val_data = (val_data.map(load_and_preprocess_image)
           .batch(batch_size)
	   .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
           .repeat()
           )
	return train_data, len(train_fnames), val_data, len(val_fnames), indexes_to_names, names_to_indexes

def main(data_folder):
	batch_size = 128
	lr = 0.0002
	# prapare dataset
	train_data, n_train, val_data, n_val, indexes_to_names, names_to_indexes = \
		prepare_datasets(data_folder, batch_size)
	# load pre-trained model
	base_model = tf.keras.applications.MobileNetV2(
    							input_shape=(128,128,3),
    							include_top=False,
    							weights='imagenet'
							)
	# Set the whole model to be trainable for simplicity
	base_model.trainable = True 
	# add a last layer for this particular classification task
	maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
	prediction_layer = tf.keras.layers.Dense(len(indexes_to_names), activation='softmax')
	model = tf.keras.Sequential([
    		base_model,
    		maxpool_layer,
    		prediction_layer
	])
	# compile model
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['sparse_categorical_accuracy']
	) 
	# run trainning procedure
	num_epochs = 5
	steps_per_epoch = round(n_train)//batch_size
	val_steps = round(n_val)//batch_size
	history = model.fit(train_data,
          epochs = num_epochs,
          steps_per_epoch = steps_per_epoch,
          validation_data=val_data, 
          validation_steps =  val_steps
	)
	# plot accuracy and loss graphs
	plt.plot(history.history['sparse_categorical_accuracy'])
	plt.plot(history.history['val_sparse_categorical_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('train_test_acc.png')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig('train_test_loss.png')

	

if __name__ == '__main__':
	if len(sys.argv) == 1:
		data_folder = Path('data/CASIA-WebFace')
	else:
		data_folder = Path(sys.argv[1])
	main(data_folder)
