# resnet18
cd ..
# 测试浮点推理
# python acc_test.py --model resnet18 --test_dataset_path ~/data/ImageNet/val

# 量化范围的准确率测试
for range in {0..224..16}
do
    echo "Reduce range: $range"

    # Quantize model
    # python quantize_model.py --reduce_range $range --model resnet18 --test_dataset_path ~/data/ImageNet/val

    # Test accuracy
    python acc_test.py --reduce_range $range --model resnet18 --test_dataset_path ~/data/ImageNet/val

    echo "**********************************************************"
done