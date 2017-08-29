# training
python train_autoencoder.py ../datasets/cam1_person001 --gpu 0 --batchsize 32 --epoch 100 --unit 100

# evaluating
python calc_score.py ../datasets/cam1_person002/  result/100.npz --unit 100
