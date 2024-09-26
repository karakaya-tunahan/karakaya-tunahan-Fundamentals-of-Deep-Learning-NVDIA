import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape, num_classes):
    base_model = keras.applications.VGG16(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

def compile_model(model):
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

def train_model(model, train_generator, valid_generator, epochs):
    model.fit(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=valid_generator.samples // valid_generator.batch_size,
        epochs=epochs
    )

def main():
    input_shape = (224, 224, 3)
    num_classes = 6
    batch_size = 16  # Reduce batch size
    epochs = 20
    
    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    datagen_valid = ImageDataGenerator(
        rescale=1.0 / 255
    )
    
    train_generator = datagen_train.flow_from_directory(
        'path_to_training_data',
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size
    )
    
    valid_generator = datagen_valid.flow_from_directory(
        'path_to_validation_data',
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size
    )
    
    with tf.device('/CPU:0'):  # Train on CPU
        model = build_model(input_shape, num_classes)
        compile_model(model)
        
        train_model(model, train_generator, valid_generator, epochs)
        
        evaluation = model.evaluate(valid_generator, steps=valid_generator.samples // valid_generator.batch_size)
        loss = evaluation[0]
        accuracy = evaluation[1]
        print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')
        
        from run_assessment import run_assessment
        run_assessment(model, valid_generator)

if __name__ == "__main__":
    main()
