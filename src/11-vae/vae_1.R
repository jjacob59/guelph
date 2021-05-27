if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()


library(keras)
library(bmp)
library(imager)
library(tidyverse)
library(abind)
library(ggpubr)

K <- keras::backend()

exp = "exp27"
numba = "_exp27_"
thresh = 0.25

# Paths ----------------------


train_events <- paste0("/home/data/refined/deep-microscopy/vaes/train_object__",  exp)
val_events <- paste0("/home/data/refined/deep-microscopy/vaes/val_object__",  exp)
test_events <- paste0("/home/data/refined/deep-microscopy/vaes/test_object__",  exp, "_test__thresh_", thresh)
  

keras_model_dir <-  file.path("/home/data/refined/deep-microscopy/vaes/keras_models", exp)
output <- file.path("/home/data/refined/deep-microscopy/vaes/", exp)

classes <- c("Yeast White" ,   "Budding White",  "Yeast Opaque",   "Budding Opaque",
             "Yeast Gray",     "Budding Gray",  
             "Artifact",       "Unknown",        "P-junction",     
             "Shmoo",          "Pseudohyphae",   "Hyphae",        
             "H-Start",        "P-Start",        "H-junction"   )

train_frac <- 1.0
val_frac <- 1.0
test_frac <- 1.0

# input image dimensions
img_rows <- 128L
img_cols <- 128L
# color channels (1 = grayscale, 3 = RGB)
img_chns <- 1L

# Data preparation --------------------------------------------------------


image_tensor <- function( targets ) {
  tmp <-  lapply(targets, FUN = function(t) {
    tmp <- load.image(t) %>% as.array
    return( array_reshape(tmp[,,1,1], c(128, 128, 1), order = "F" ) )
  }) 
  return( abind( tmp, along = 0 ) )
}

get_labels <- function( fs ) {
  fs_1 <- str_split(fs, pattern = "/")
  fs_2 <-  lapply(fs_1, "[[", 8)
  fs_3 <- lapply(str_split( fs_2, pattern=".bmp"), "[[", 1)
  fs_4 <- as.numeric( unlist( lapply(str_split(fs_3, pattern = "_"), "[[", 8) ) )
  fs_5 <- unlist(lapply( fs_4, FUN = function(f) return( classes[f+1] )))
  return( fs_5 )
} 

get_test_labels <- function( fs ) {
  fs_1 <- str_split(fs, pattern = "/")
  fs_2 <-  lapply(fs_1, "[[", 8)
  fs_3 <- lapply(str_split( fs_2, pattern=".bmp"), "[[", 1)
  fs_4 <- unlist( lapply(str_split(fs_3, pattern = "_"), "[[", 2) ) 
  return( fs_4 )
} 

train_fs <- list.files(train_events, full.names = TRUE)
files_x_train <- sample(train_fs, length(train_fs) * train_frac, replace = FALSE)
labels_x_train <- get_labels(files_x_train)

val_fs<- list.files(val_events, full.names = TRUE)
files_x_val <- sample(val_fs, length(val_fs) * val_frac, replace = FALSE)
labels_x_val <- get_labels(files_x_val)

test_fs <- list.files(test_events, full.names = TRUE)
files_x_test <- sample(test_fs, length(test_fs) * test_frac, replace = FALSE)
labels_x_test <- get_test_labels(files_x_test)

x_train <- image_tensor( files_x_train ) 
x_val  <- image_tensor( files_x_val )
x_test <- image_tensor( files_x_test )

cat("\n Dimensions of train: ", dim(x_train), 
    "\t Dimensions of val: ", dim(x_val), 
    "\t Dimensions of test: ", dim(x_test), "\n")

#### Parameterization ####


# number of convolutional filters to use
filters <- 64L

# convolution kernel size
num_conv <- 5L

latent_dim <- 3L
intermediate_dim <-  128L
epsilon_std <- 1.0

# training parameters
batch_size <- 100L
eps <- 10L


#### Model Construction ####

original_img_size <- c(img_rows, img_cols, img_chns)

x <- layer_input(shape = c(original_img_size))

conv_1 <- layer_conv_2d(  #conv2d_20
  x,
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_2 <- layer_conv_2d(  #convd_21
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d( #convd_22
  conv_2,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d( #convd_23
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")

z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

#output_shape <- c(batch_size, 14L, 14L, filters)
#output_shape <- c(batch_size, 64L, 64L, filters)
output_shape <- c(batch_size, 64L, 64L, filters)


decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = 64,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "valid",
  activation = "relu"
)

decoder_mean_squash <- layer_conv_2d(
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "valid",
  activation = "sigmoid"
)

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)

# custom loss function
vae_loss <- function(x, x_decoded_mean_squash) {
  x <- k_flatten(x)
  x_decoded_mean_squash <- k_flatten(x_decoded_mean_squash)
  xent_loss <- 1.0 * img_rows * img_cols *
    loss_binary_crossentropy(x, x_decoded_mean_squash)
  kl_loss <- -0.5 * k_mean(1 + z_log_var - k_square(z_mean) -
                             k_exp(z_log_var), axis = -1L)
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash)
#vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% compile(optimizer = optimizer_adam(), loss = vae_loss)  # better than rmsprop
#vae %>% compile(optimizer = optimizer_nadam(), loss = vae_loss)  # not bad. a bit compressed. compare with adam
#vae %>% compile(optimizer = optimizer_adagrad(), loss = vae_loss)  # also in contention. A bit compressed
#vae %>% compile(optimizer = optimizer_adadelta(), loss = vae_loss) # also good


optimiza <- "adam"

summary(vae)

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## build a digit generator that can sample from the learned distribution
gen_decoder_input <- layer_input(shape = latent_dim)
gen_hidden_decoded <- decoder_hidden(gen_decoder_input)
gen_up_decoded <- decoder_upsample(gen_hidden_decoded)
gen_reshape_decoded <- decoder_reshape(gen_up_decoded)
gen_deconv_1_decoded <- decoder_deconv_1(gen_reshape_decoded)
gen_deconv_2_decoded <- decoder_deconv_2(gen_deconv_1_decoded)
gen_x_decoded_relu <- decoder_deconv_3_upsample(gen_deconv_2_decoded)
gen_x_decoded_mean_squash <- decoder_mean_squash(gen_x_decoded_relu)
generator <- keras_model(gen_decoder_input, gen_x_decoded_mean_squash)



#### Model Fitting ####


with(tensorflow::tf$device('GPU:8'), {
  vae %>% fit(
    x_train, x_train, 
    shuffle = TRUE, 
    epochs = eps, 
    batch_size = batch_size, 
    validation_data = list(x_val, x_val)
  )
})

gc()

#vae %>% model.save((file.path(save_keras_dir, "first_time"))
#vae %>% export_savedmodel(file.path(save_keras_dir, "first_time"), remove_learning_phase = FALSE)
vae %>% save_model_weights_tf(keras_model_dir)

                   

#### Visualizations ####

library(ggplot2)
library(dplyr)
library(pals)


# -> prep the test1 and test2 tibbles for umap

x_train_encoded <- predict(encoder, x_train, batch_size = batch_size)
x_val_encoded <- predict(encoder, x_val, batch_size = batch_size)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)


x_train_encoded <- x_train_encoded %>% as_data_frame() %>% 
  mutate(class = as.factor(labels_x_train))
x_train_encoded %>% 
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +  scale_color_manual(values = watlington())
x_train_encoded[['type']] <- "train"

x_val_encoded <- x_val_encoded %>% as_data_frame() %>% 
  mutate(class = as.factor(labels_x_val))
x_val_encoded %>% 
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +  scale_color_manual(values = watlington())
x_val_encoded[['type']] <- "val"

x_test_encoded <- x_test_encoded %>% as_data_frame() %>%
  mutate(class = as.factor(labels_x_test))
x_test_encoded %>% 
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +  scale_color_manual(values = watlington())
x_test_encoded[['type']] <- "test"

x_all_encoded <- bind_rows( x_val_encoded, x_test_encoded )
x_all_encoded <- bind_rows( x_all_encoded, x_train_encoded )
x_all_encoded <- x_all_encoded %>% mutate( class_pos = as.numeric(class)  )
saveRDS(x_all_encoded, file = paste0("~/vae_all_", 2, ".Rdata"))

# <------ umap

library(umap)

# x <- umap( x_all_encoded %>% select( starts_with('V'))) 

xte <- x_all_encoded 
g1 <- xte %>% ggplot(aes(x = V1, y = V2, color = class) ) +
  geom_point(size = 1) +
  scale_color_manual( values = watlington())
g2 <-   xte %>% ggplot(aes(x = V1, y = V3, color = class) ) +
  geom_point(size = 1) +
  scale_color_manual( values = watlington())
g3 <- xte %>% ggplot(aes(x = V2, y = V3, color = class) ) +
  geom_point(size = 1) +
  scale_color_manual( values = watlington())

ggarrange(g1, g2, g3, ncol=3)




