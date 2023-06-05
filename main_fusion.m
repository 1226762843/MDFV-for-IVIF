
close all;
clear all;
clc;

addpath(genpath(strcat(pwd,'\','fusion_metrix')));

dataset ={'Dataset_TNO'};
Fcell = {'l2-norm'}; 
R_gradcell = {'TV-max','2FTV-max','4FTV-max','8FTV-max'};
R_lumicell = {'Z-max'}; 
Fraccell = {'AL2'}; 
num_model = 0; 
WT = table;

for i_dataset = 1 : length(dataset)
    clear metri;
    present_dataset = [dataset{i_dataset},'.mat'];
    disp(['Dataset : ' present_dataset]);
    load(present_dataset);
    for i1 = 1 : length(Fcell)
        F = Fcell{i1};
        for i2 = 1 : length(R_gradcell)
            R_grad = R_gradcell{i2};
            switch R_grad
                case 'TV-max'
                    v = 1;
                    ilam = 2; 
                    ibeta = 18;
                case '2FTV-max'
                    v = 1.3;
                    ilam = 2; 
                    ibeta = 18;
                case '4FTV-max'
                    v = 1;
                    ilam = 3; 
                    ibeta = 20;
                case '8FTV-max'
                    v = 1.2;
                    ilam =0.2; 
                    ibeta = 19.5;
            end
            for i3 = 1 : length(R_lumicell)
                R_lumi = R_lumicell{i3};
                for i4 = 1 : length(Fraccell)
                    Frac = Fraccell{i4};
                    for i5 = 1 : length(ilam)
                        for i6 = 1 : length(ibeta)
                            lam = ilam(i5);
                            beta = ibeta(i6);
                            K_gus = 0.4;
                            K = fspecial('gaussian', 3 ,K_gus);
                            dt = 1;  
                            iter = 500;
                            resultpath = ['.\result_TNO\',R_grad,'\',];

                            switch Frac
                                case 'AL2'
                                    cs_1 = v/4+(v^2)/8;
                                    cs0  = 1-(v^2)/2-(v^3)/8;
                                    cs1  = -5*v/4+5*(v^3)/16+(v^4)/16;
                                case 'GL'
                                    cs_1 = 1;
                                    cs0  = -v;
                                    cs1  = (v^2-v)/2;
                                case 'RL'
                                    cs_1 = 1/(gamma(-v)*(-2*v)+eps);
                                    cs0  = (2^(-v))/(gamma(-v)*(-2*v)+eps);
                                    cs1  = (3^(-v)-1^(-v))/(gamma(-v)*(-2*v)+eps);
                            end

                            mask_al2_xnegative  = [0 0 0;cs1 cs0 cs_1;0 0 0]; % x- direcion
                            mask_al2_xpostive = [0 0 0;cs_1 cs0 cs1;0 0 0]; % x+ direcion
                            mask_al2_ynegative = [0 cs_1 0;0 cs0 0;0 cs1 0]; % y- direcion
                            mask_al2_ypostive  = [0 cs1 0;0 cs0 0;0 cs_1 0]; % y+ direcion
                            mask_al2_RD = [cs_1 0 0;0 cs0 0; 0 0 cs1]; % 
                            mask_al2_LU = [cs1 0 0;0 cs0 0; 0 0 cs_1]; % 
                            mask_al2_LD = [0 0 cs_1;0 cs0 0; cs1 0 0 ]; % 
                            mask_al2_RU = [0 0 cs1;0 cs0 0; cs_1 0 0]; % 

                            mask2_x = mask_al2_xpostive;
                            mask2_y = mask_al2_ypostive;

                            mask4_x1 = mask_al2_xpostive;
                            mask4_x2 = mask_al2_xnegative;
                            mask4_y1 = mask_al2_ypostive;
                            mask4_y2 = mask_al2_ynegative;

                            mask8_x1 = mask_al2_xpostive; mask8_x2 = mask_al2_xnegative;
                            mask8_y1 = mask_al2_ypostive; mask8_y2 = mask_al2_ynegative;
                            mask8_LU = mask_al2_LU; mask8_RD = mask_al2_RD;
                            mask8_RU = mask_al2_RU; mask8_LD = mask_al2_LD;

                            mask5x5 = [cs1 0 cs1 0 cs1; 0 cs0 cs0 cs0 0; cs1 cs0 8*cs_1 cs0 cs1;...
                                0 cs0 cs0 cs0 0;cs1 0 cs1 0 cs1]/(8*(cs1+cs0+cs_1)+eps);

                            for im = 1:i_pairs
                                num_model = num_model +1;
                                Dataset{num_model} = dataset{i_dataset};
                                momentum = 0.9;
                                F_momentum = 0;
                                disp(['Pairs : ' num2str(im)]);
                                U = double(image_sets{im,1});
                                V = double(image_sets{im,2});
                                W1 = 1; W2=1;   
                                Z = (U+V)/2;
                                tic
                                for it=1:iter
                                    %% Fidelity
                                    switch F
                                        case 'l2-norm'  % F1  l2-norm
                                            F_Z = W1.*(Z-U) + W2.*(Z-V) ;
                                        otherwise
                                            warning('NO EXISTING NORM')
                                    end
                                    %% Regularization-gradient
                                    if lam ~=0
                                        switch R_grad
                                            case 'TV-max'
                                                [ Vx , Vy ] = gradient(Z-V);
                                                [ Ux , Uy ] = gradient(Z-U);
                                                Mx = min(Vx,Ux);My = min(Vy,Uy);
                                                Z_grad = sqrt( Mx.^2 + My.^2 ) + eps;
                                                [ Mxx , Myy ] = gradient(Z_grad);
                                                F_Z = F_Z + lam .*  ( Mxx + Myy )./Z_grad;
                                            case '2FTV-max'
                                                Vx = conv2((Z-V),mask2_x,'same');
                                                Vy = conv2((Z-V),mask2_y,'same');
                                                Ux = conv2((Z-U),mask2_x,'same');
                                                Uy = conv2((Z-U),mask2_y,'same');
                                                Mx = min(Vx,Ux);My = min(Vy,Uy);
                                                DEN = sqrt( Mx.^2 + My.^2 ) + eps;
                                                Mxx = conv2(Mx./DEN,mask2_x,'same');
                                                Myy = conv2(My./DEN,mask2_y,'same');
                                                F_Z = F_Z + lam .*  ( Mxx + Myy );
                                            case '4FTV-max'
                                                Vx1 = conv2((Z-V),mask4_x1,'same');
                                                Vy1 = conv2((Z-V),mask4_y1,'same');
                                                Vx2 = conv2((Z-V),mask4_x2,'same');
                                                Vy2 = conv2((Z-V),mask4_y2,'same');
                                                Ux1 = conv2((Z-U),mask4_x1,'same');
                                                Uy1 = conv2((Z-U),mask4_y1,'same');
                                                Ux2 = conv2((Z-U),mask4_x2,'same');
                                                Uy2 = conv2((Z-U),mask4_y2,'same');
                                                Mx1 = min(Vx1,Ux1);My1 = min(Vy1,Uy1);
                                                Mx2 = min(Vx2,Ux2);My2 = min(Vy2,Uy2);
                                                DEN = sqrt( Mx1.^2 + My1.^2 + Mx2.^2 + My2.^2) + eps;
                                                Mxx1 = conv2(Mx1./DEN,mask4_x1,'same');
                                                Myy1 = conv2(My1./DEN,mask4_y1,'same');
                                                Mxx2 = conv2(Mx2./DEN,mask4_x2,'same');
                                                Myy2 = conv2(My2./DEN,mask4_y2,'same');
                                                F_Z = F_Z + lam .*  ( Mxx1 + Myy1 + Mxx2 + Myy2 );
                                            case '8FTV-max'
                                                Vx1 = conv2((Z-V),mask8_x1,'same');
                                                Vy1 = conv2((Z-V),mask8_y1,'same');
                                                Vx2 = conv2((Z-V),mask8_x2,'same');
                                                Vy2 = conv2((Z-V),mask8_y2,'same');
                                                V_LU = conv2((Z-V),mask8_LU,'same');
                                                V_RD = conv2((Z-V),mask8_RD,'same');
                                                V_RU = conv2((Z-V),mask8_RU,'same');
                                                V_LD = conv2((Z-V),mask8_LD,'same');
                                                Ux1 = conv2((Z-U),mask8_x1,'same');
                                                Uy1 = conv2((Z-U),mask8_y1,'same');
                                                Ux2 = conv2((Z-U),mask8_x2,'same');
                                                Uy2 = conv2((Z-U),mask8_y2,'same');
                                                U_LU = conv2((Z-U),mask8_LU,'same');
                                                U_RD = conv2((Z-U),mask8_RD,'same');
                                                U_RU = conv2((Z-U),mask8_RU,'same');
                                                U_LD = conv2((Z-U),mask8_LD,'same');
                                                Mx1 = min(Vx1,Ux1);My1 = min(Vy1,Uy1);
                                                Mx2 = min(Vx2,Ux2);My2 = min(Vy2,Uy2);
                                                M_LU = min(V_LU,U_LU);M_RD = min(V_RD,U_RD);
                                                M_RU = min(V_RU,U_RU);M_LD = min(V_LD,U_LD);
                                                DEN = sqrt( Mx1.^2 + My1.^2 + Mx2.^2 + My2.^2 + ...
                                                    M_LU.^2 + M_RD.^2 + M_RU.^2 + M_LD.^2) + eps;
                                                Mxx1 = conv2(Mx1./DEN,mask8_x1,'same');
                                                Myy1 = conv2(My1./DEN,mask8_y1,'same');
                                                Mxx2 = conv2(Mx2./DEN,mask8_x2,'same');
                                                Myy2 = conv2(My2./DEN,mask8_y2,'same');
                                                M_LU1 = conv2(M_LU,mask8_LU,'same');
                                                M_RD1 = conv2(M_RD,mask8_RD,'same');
                                                M_RU1 = conv2(M_RU,mask8_RU,'same');
                                                M_LD1 = conv2(M_LD,mask8_LD,'same');
                                                F_Z = F_Z + lam .*  ( Mxx1 + Myy1 + Mxx2 + Myy2 + M_LU1 + M_RD1 + M_RU1 + M_LD1 );
                                            otherwise
                                                warning('NO EXISTING Regularization-gradient')
                                        end   
                                    end
                                    %% Regularization-luminance
                                   if beta ~= 0
                                        switch R_lumi
                                            case 'Z-U-V'
                                                F_Z = F_Z + beta .* (Z-U-V);
                                            case 'Z'
                                                F_Z = F_Z - beta .* (Z);
                                            case 'Z-max'
                                                F_Z = F_Z + beta .* (Z-255*max(U./max(U(:)),V./max(V(:))));
                                            otherwise
                                                warning('NO EXISTING Regularization-luminance')
                                        end
                                    end
                                    F_t = F_Z ;
                                    F_momentum = momentum.*F_momentum + (1-momentum).*F_t;
                                    Z = Z - dt * F_momentum ; 
                                    Z(Z>255)=255;
                                    Z(Z<0)=0;
                                end
                                toc
                                if isfolder(resultpath)==0
                                    mkdir(resultpath);
                                end
                                namepath = [F,'-lam=',num2str(lam),' ',R_grad,'-beta=',num2str(beta),' ',R_lumi];
                                savepath = [resultpath,namepath];
                                imgname{num_model} = namepath;

                                imwrite(Z/255,[savepath,num2str(im),'.tif']);
                                savepathUVZ = [resultpath,'[UVZ]--',namepath];

                                metri_name = {'pairs','LAM','BETA','Order','MIN','Qsi','QABF','VIFP'};
                                num_metri = length(metri_name);
                                for ii=1:num_metri
                                    metri(1,ii) = fusion_metrix(image_sets{im,1},image_sets{im,2},uint8(Z),metri_name{ii});
                                    metri(1,1) = im;
                                    metri(1,2) = lam;
                                    metri(1,3) = beta;
                                    metri(1,4) = v;
                                end
                                A(num_model,:) = metri;
                                % A{i_dataset}(im,:) = metri;
                                % T{i_dataset}{im} = array2table(metri,'VariableNames',metri_name);
                                disp(['Model : ',imgname{num_model}]);
                            end
                        end
                    end
                end
            end
        end
    end
    Pairs = A(:,1);
    LAM = A(:,2);
    BETA = A(:,3);
    Order = A(:,4);
    MIN = A(:,5);
    QY = A(:,6);
    QABF = A(:,7);
    VIFP = A(:,8);
    Model = imgname';

    WT = table(Pairs,LAM,Order,MIN,QY,QABF,VIFP);

    writetable(WT,[resultpath,F,'-lam=',num2str(ilam(1)),'to',num2str(ilam(length(ilam))),' ',R_grad,...
        '-beta=',num2str(ibeta(1)),'to',num2str(ibeta(length(ibeta))),' ',R_lumi,'.xlsx'],'WriteRowNames',true,'WriteVariableNames',true);
end
%writetable(WT,savepath,'WriteRowNames',true,'WriteVariableNames',false,'Sheet','MIN');
% Interval_matrix = ones(1,5)*99999.9999;
% hebing = [A{1};Interval_matrix;A{2};Interval_matrix;...
%     A{3};Interval_matrix;A{4};Interval_matrix;A{5};...
%     Interval_matrix;A{6};Interval_matrix;A{7}]';
% result_all = [hebing,mean(hebing(:,2:5),2)];





















