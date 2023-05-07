
for layer in resnet152V2.layers:
    layer.trainable = False


folders = glob('PlantVillage1/train/*')

resnet152V2.output


x = Flatten()(resnet152V2.output)


len(folders)


prediction = Dense(len(folders), activation='softmax')(x)


model = Model(inputs = resnet152V2.input, outputs = prediction)

model.summary()

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),loss='categorical_crossentropy', metrics=["accuracy"])


def fitness_function(params):
    learning_rate = params[0]
    batch_size = int(params[1])
    epochs = int(params[2])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    r = model.fit_generator(train_set, 
                            validation_data=test_set, 
                            epochs=epochs, 
                            steps_per_epoch=len(train_set), 
                            validation_steps=len(test_set),
                            batch_size=batch_size,
                            verbose=0)
    accuracy = r.history['accuracy']
    return  accuracy
