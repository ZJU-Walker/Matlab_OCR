clc;
clear all;
close all;
%%
%图像预处理（转化成灰度和bw图）
%读取一张图片，并显示
original_picture=imread('IMG/test3.png');
figure(1);
imshow(original_picture);
title('原始RGB图像')
%把图像转换成灰度图像
GrayPic=rgb2gray(original_picture);%把RGB图像转化成灰度图像
figure(2) 
imshow(GrayPic);
title('RGB图像转化为灰度图像')
%对图像进行二值化处理
%将graypic转化为bw图
thresh=graythresh(GrayPic);
BW_pic=im2bw(GrayPic,thresh);
figure(3);
imshow(BW_pic),hold on
saveas(figure(3),'BW_pic.png')
title('灰度图像转化为二值化图像')
%%
L = bwlabel(BW_pic);%标记连通区域
stats = regionprops(L);
Ar = cat(1, stats.Area);
ind = find(Ar ==max(Ar));%找到最大连通区域的标号
BW_pic(find(L~=ind))=0;%将其他区域置为0
%裁剪
Bound_all = cat(1,stats.BoundingBox)%获取所有区域矩形
Bound = Bound_all(ind,:)%找到最大区域包围矩形
rectangle('Position',Bound, 'EdgeColor','r', 'LineWidth',2)
Gray_cut = imcrop(GrayPic,Bound);
BW_cut = imcrop(BW_pic,Bound);
figure(4);
% img=Gray_cut
img=BW_cut;
imshow(img);
title('请依次选取PPT区域的左上、右上、左下、右下四个点,选择完毕后请按shift键')
saveas(figure(4),'BW_cut.png')
[m,n] = size(img);
dot=ginput(4);       %取四个点，依次是左上，右上，左下，右下
dot(:,[1,2])= dot(:,[2,1]); % 变换x y坐标，x=行 y=列
col=round(sqrt((dot(1,1)-dot(2,1))^2+(dot(1,2)-dot(2,2))^2));   %从原四边形获得新矩形宽
row=round(sqrt((dot(1,1)-dot(3,1))^2+(dot(1,2)-dot(3,2))^2));   %从原四边形获得新矩形高
new_img = ones(row,col);
% 原图四个基准点的坐标
x = [dot(1,1),dot(2,1),dot(3,1),dot(4,1)];
y = [dot(1,2),dot(2,2),dot(3,2),dot(4,2)];
% 新图四个基准点坐标
X = [1,1,row,row];
Y = [1,col,1,col];
% 列出投影关系 求出投影矩阵
A=[x(1),y(1),1,0,0,0,-X(1)*x(1),-X(1)*y(1);             
   0,0,0,x(1),y(1),1,-Y(1)*x(1),-Y(1)*y(1);
   x(2),y(2),1,0,0,0,-X(2)*x(2),-X(2)*y(2);
   0,0,0,x(2),y(2),1,-Y(2)*x(2),-Y(2)*y(2);
   x(3),y(3),1,0,0,0,-X(3)*x(3),-X(3)*y(3);
   0,0 ,0,x(3),y(3),1,-Y(3)*x(3),-Y(3)*y(3);
   x(4),y(4),1,0,0,0,-X(4)*x(4),-X(4)*y(4);
   0,0,0,x(4),y(4),1,-Y(4)*x(4),-Y(4)*y(4)];%求解变换矩阵的行列式
B = [X(1),Y(1),X(2),Y(2),X(3),Y(3),X(4),Y(4)]';
C = inv(A)*B;
D = [C(1),C(2),C(3);
     C(4),C(5),C(6);
     C(7),C(8),1]; % 变换矩阵3*3模式
 inv_D = inv(D);
 for i = 1:row
     for j = 1:col
       % 解二元一次方程组，根据目标图像坐标反求出原图坐标
       pix = inv_D * [i j 1]';
       pix1 = inv([C(7)*pix(1)-1 C(8)*pix(1);C(7)*pix(2) C(8)*pix(2)-1])*[-pix(1) -pix(2)]';
       if pix1(1)<m && pix1(2)<n
           new_img(i,j) = img(round(pix1(1)),round(pix1(2))); %最近邻插值
       else
           new_img(i,j) = 255;
       end
     end
 end
         
figure(10);
imshow(new_img,[]);
saveas(figure(10),'After.png')
img = imread('After.png');
text=ocr(new_img).Text %Ocr识别


load('net_best__2022_06_18__23_43_50.mat');%加载预训练神经网络
encChinese = netBest.encGerman;
encEnglish = netBest.encEnglish;
netEncoder = netBest.netEncoder;
netDecoder = netBest.netDecoder;
text = convertCharsToStrings(text);
strEngNew = [text];
strTranslatedNew = translateText(netEncoder,netDecoder,encChinese,encEnglish,strEngNew)