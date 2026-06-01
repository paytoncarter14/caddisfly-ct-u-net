library(tidyverse)
library(png)
library(keras)

img_width = 256
img_height = 256

# Function to resize matrix with nearest neighbor
resizem = function(M, rows, cols) {
    rs = round(seq(0, rows-1)/(rows-1) * (nrow(M)-1) +1)
    cs = round(seq(0, cols-1)/(cols-1) * (ncol(M)-1) +1)
    M[rs, ][, cs]
}

file_list = sample(list.files('cropped'))

prepare_array = function(folder) {

    output = c()

    for (file in file_list) {
        # Read png into matrix. Images aren't saved as grayscale but they are. All three channels have equal values, so just keep the first channel.
        png = readPNG(paste0(folder, '/', file))[,,1]

        # Resize with nearest neighbor
        resized = resizem(png, img_height, img_width)

        output = c(output, resized)
    }

    output = aperm(array(output, dim=c(256, 256, 144)))

}

cropped_array = prepare_array('cropped')
# make black background = mean
for (ii in 1:dim(cropped_array)[1]) {
    slice_mean = mean(cropped_array[ii,,])
    cropped_array[ii,,] = ifelse(cropped_array[ii,,] < 0.01, slice_mean, cropped_array[ii,,])
}
# scale to 0 through 1
for (ii in 1:dim(cropped_array)[1]) {
    slice_min = min(cropped_array[ii,,])
    slice_range = diff(range(cropped_array[ii,,]))
    cropped_array[ii,,] = (cropped_array[ii,,] - slice_min) / slice_range
}

# split training/test data
traced_array = prepare_array('traced')

cropped_array_train = prepare_array('cropped')[1:120,,]
traced_array_train = prepare_array('traced')[1:120,,]

cropped_array_test = prepare_array('cropped')[121:144,,]
traced_array_test = prepare_array('traced')[121:144,,]

early_stop = callback_early_stopping(
  monitor = "val_loss",
  patience = 20,
  restore_best_weights = TRUE
)

input = layer_input(shape = c(256, 256, 1))

# model definition with concat layers
# encoder
enc_1 = input %>% # enc_1 = 128x128
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = 'same') %>%
    layer_activation_leaky_relu() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_batch_normalization()

enc_2 = enc_1 %>% # enc_2 = 64x64
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = 'same') %>%
    layer_activation_leaky_relu() %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_batch_normalization()

# bottleneck
bottleneck = enc_2 %>% # bottleneck = 64x64
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = 'same') %>%
    layer_activation_leaky_relu() %>%
    layer_batch_normalization()

concat_1 = layer_concatenate(inputs = list(enc_2, bottleneck))

# decoder
dec_1 = concat_1 %>% # dec_1 = 128x128
    layer_conv_2d(filters = 16, kernel_size = c(3, 3), padding = 'same') %>%
    layer_activation_leaky_relu() %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_batch_normalization()

concat_2 = layer_concatenate(inputs = list(enc_1, dec_1))

dec_2 = concat_2 %>% # dec_2 = 256x256
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), padding = 'same') %>%
    layer_activation_leaky_relu() %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    layer_batch_normalization()

# output
output = dec_2 %>% # output = 256x256
    layer_conv_2d(filters = 1, kernel_size = c(1, 1), activation = 'sigmoid')

model = keras_model(inputs = input, outputs = output)

# compile and train
compile(
    model,
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = 'accuracy'
)

history = model %>% fit(
    x = cropped_array_train,
    y = traced_array_train,
    epochs = 1000,
    batch_size = 10,
    callbacks = early_stop,
    validation_split = 0.2
)

# make test predictions
predictions = predict(model, cropped_array_test)
thresholded_predictions = ifelse(predictions >= 0.3, 1, 0)

# save test predictions as png
save_output = function(name, var) {
    for (ii in 1:24) {
        png(paste0('output/', ii, '_', name, '.png'))
        plot(as.raster(var[ii,,]), axes=F, box=F, bty='n')
        dev.off()
    }
}

save_output('predict', thresholded_predictions[,,,])
save_output('traced', traced_array_test)
save_output('cropped', cropped_array_test)