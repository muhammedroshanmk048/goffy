train_path = 'PlantVillage1/train'
test_path = 'PlantVillage1/val'

image_size = [224,224]

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.2,
                                   shear_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


train_set = train_datagen.flow_from_directory('PlantVillage1/train',
                                              target_size=(224, 224),
                                              batch_size = 32,
                                              class_mode = 'categorical')
                                          
test_set = test_datagen.flow_from_directory('PlantVillage1/val',
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
