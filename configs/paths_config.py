dataset_paths = {
	'church_train': '/nfs/datasets/segmentation/lsun/lsun/church_outdoor_train',
	'church_test': '/nfs/datasets/segmentation/lsun/lsun/church_outdoor_val',
	'tower_train': '/nfs/datasets/segmentation/lsun/lsun/tower_train/',
	'tower_test': '/nfs/datasets/segmentation/lsun/lsun/tower_val/',
	'cars_train': '/nfs/datasets/segmentation/stanford_cars/cars_train',
	'cars_test': '/nfs/datasets/segmentation/stanford_cars/cars_test',
	'celeba_train': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/train_img',
	'celeba_test': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/test_img',
	'celeba_train_sketch': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/train_img_edges_256/',
	'celeba_test_sketch': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/test_img_edges_256/',
	'celeba_train_segmentation': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/train_mask/',
	'celeba_test_segmentation': '/nfs/datasets/segmentation/celebAMask/CelebAMask-HQ/test_mask/',
	# 'ffhq': '/nfs/datasets/segmentation/ffhq/images256x256',
	'ffhq': '../../input/ffhq-256x256/images256x256/',
	'ffhq_unaligned': '/nfs/datasets/segmentation/ffhq/unaligned1024x1024/',
	'ffhq_toonify': '/nfs/datasets/segmentation/ffhq/images256x256_toonified',
	'afhq_cat_train': '/nfs/datasets/segmentation/afhq/train/cat',
	'afhq_cat_test': '/nfs/datasets/segmentation/afhq/val/cat',
	'afhq_dog_train': '/nfs/datasets/segmentation/afhq/train/dog',
	'afhq_dog_test': '/nfs/datasets/segmentation/afhq/val/dog',
	'afhq_wild_train': '/nfs/datasets/segmentation/afhq/train/wild',
	'afhq_wild_test': '/nfs/datasets/segmentation/afhq/val/dog',
	'afhq_cat_train_sketch': '/nfs/datasets/segmentation/afhq/train/cat_edges',
	'afhq_cat_test_sketch': '/nfs/datasets/segmentation/afhq/val/cat_edges',
	'afhq_dog_train_sketch': '/nfs/datasets/segmentation/afhq/train/dog_edges',
	'afhq_dog_test_sketch': '/nfs/datasets/segmentation/afhq/val/dog_edges',

	'ffhq_unaligned_landmarks_transforms': '/nfs/private/yuval/or/stylegan3-paper/inversion/ffhq_dict.pickle',
	'62k_ffhq_unaligned_landmarks_transforms': '/nfs/private/yuval/or/stylegan3-paper/inversion/62k_train_ffhq_dict.pickle',
	'1k_ffhq_unaligned_landmarks_transforms': '/nfs/private/yuval/or/stylegan3-paper/inversion/1k_test_ffhq_dict.pickle'
}

model_paths = {
	# models for backbones and losses
	'ir_se50': '/nfs/weights/pix2profile/model_ir_se50.pth',
	'resnet34': '/nfs/weights/pix2profile/resnet34-333f7ec4.pth',
	'moco': '/nfs/weights/pix2profile/moco_v2_800ep_pretrain.pt',
	# stylegan2 generators
	'stylegan_ffhq': '/nfs/weights/pix2profile/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': 'pretrained_models/stylegan2-car-config-f.pt',
	'stylegan_ada_wild': '/nfs/weights/pix2profile/afhqwild.pt',
	# stylegan3 generators
	'stylegan3_ffhq': '/nfs/weights/stylegan3/stylegan3-r-ffhq-1024x1024.pkl',
	'stylegan3_ffhq_unaligned': '/nfs/weights/stylegan3/stylegan3-r-ffhqu-1024x1024.pkl',
	'stylegan3_ffhq_pt': '/nfs/weights/stylegan3/sg3-r-ffhq-1024.pt',
	'stylegan3_ffhq_unaligned_pt': '/nfs/weights/stylegan3/sg3-r-ffhqu-1024.pt',
	# model for face alignment
	'shape_predictor': '/nfs/weights/pix2profile/shape_predictor_68_face_landmarks.dat',
	# models for ID similarity computation
	'curricular_face': '/nfs/weights/pix2profile/CurricularFace_Backbone.pth',
	'mtcnn_pnet': '/nfs/weights/pix2profile/mtcnn/pnet.npy',
	'mtcnn_rnet': '/nfs/weights/pix2profile/mtcnn/rnet.npy',
	'mtcnn_onet': '/nfs/weights/pix2profile/mtcnn/onet.npy',
	# WEncoders for training on various domains
	'faces_w_encoder': 'pretrained_models/faces_w_encoder.pt',
	'cars_w_encoder': '../pretrained_models/cars_w_encoder.pt',
	'afhq_wild_w_encoder': 'pretrained_models/afhq_wild_w_encoder.pt',
	# models for domain adaptation
	'restyle_e4e_ffhq': 'pretrained_models/restyle_e4e_ffhq_encode.pt',
	'stylegan_pixar': 'pretrained_models/pixar.pt',
	'stylegan_toonify': 'pretrained_models/ffhq_cartoon_blended.pt',
	'stylegan_sketch': 'pretrained_models/sketch.pt',
	'stylegan_disney': 'pretrained_models/disney_princess.pt',

	'align_net': '/nfs/private/yuval/or/stylegan3-paper/inversion/correct_alignment/results/first_try/checkpoints/iteration_48000.pt'

}

styleclip_directions = {
	"ffhq": {
		'delta_i_c': 'editing/styleclip/global_directions/sg3-r-ffhq-1024/delta_i_c_ffhq.npy',
		's_statistics': 'editing/styleclip/global_directions/sg3-r-ffhq-1024/stat',
	},
	'templates': 'editing/styleclip/global_directions/templates.txt'
}

stylegan3_aligned_edit_paths = {
	'age': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/age_boundary.npy',
	'smile': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Smiling_boundary.npy',
	'pose': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/pose_boundary.npy',
	'Black_Hair': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Black_Hair_boundary.npy',
	'Bald': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Bald_boundary.npy',
	'Chubby': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Chubby_boundary.npy',
	'Eyeglasses': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Eyeglasses_boundary.npy',
	'Male': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Male_boundary.npy',
	'No_Beard': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/No_Beard_boundary.npy',
	'Mustache': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Mustache_boundary.npy',
	'Goatee': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Goatee_boundary.npy',
	'Heavy_Makeup': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_aligned/Heavy_Makeup_boundary.npy',
}

stylegan3_unaligned_edit_paths = {
	'age': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/age_boundary.npy',
	'smile': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Smiling_boundary.npy',
	'pose': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/pose_boundary.npy',
	'Black_Hair': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Black_Hair_boundary.npy',
	'Bald': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Bald_boundary.npy',
	'Chubby': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Chubby_boundary.npy',
	'Eyeglasses': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Eyeglasses_boundary.npy',
	'Heavy_Makeup': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Heavy_Makeup_boundary.npy',
	'Male': '/nfs/outputs/stylegan3/interfacegan/boundaries/w_unaligned_with_pseudo_alignment/Male_boundary.npy',
}
















