WinCLIP 项目文件（前辈自备部分，由小陌下载代发）

一共 12 个分卷 + 本说明 + merge.bat：
- weights_part_00 到 weights_part_09：模型权重 OpenCLIP ViT-B-16-plus-240（laion400m_e32）的分卷
- transistor_part_00、transistor_part_01：测试图片包 transistor_test.zip 的分卷

用法：
1. 把 12 个分卷和 merge.bat 下载到同一个文件夹
2. 双击 merge.bat，会合并出 open_clip_pytorch_model.bin（约 795 MB）和 transistor_test.zip（约 123 MB）
3. open_clip_pytorch_model.bin 就是权重文件，按本地 AI 给的 DOWNLOAD_CHECKLIST 放到它指定的目录
4. transistor_test.zip 解压后是 transistor 文件夹（test 100 张图 + ground_truth 掩码 40 张），同样按清单放

注意：分卷少一个都合并不了；合并完可以删掉分卷省空间。
