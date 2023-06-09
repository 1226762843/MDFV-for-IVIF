function Q = Qsi_index(img1,img2,imgf,block_size)
S1 = SSIM_window(img1,imgf,block_size);
S2 = SSIM_window(img2,imgf,block_size);
lamd = lamda_compute1(img1,img2,block_size);
S3 = SSIM_window(img1,img2,block_size);
Q1 = lamd.*S1 + (1-lamd).*S2;
index = S3<0.75;
S4 = max(S1,S2);
Q1(index) = S4(index);
Q = mean2(Q1);