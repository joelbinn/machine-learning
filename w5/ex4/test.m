X=[2 3;4 5]Y=[1 0; 0 1]Theta1=[.1 .2 .3; .4 .5 .6; .7 .8 .9]Theta2=[-.1 -.2 -.3 -.4; -.5 -.6 -.7 -.8]a_1=Xa_2=sigmoid([ones(size(a_1,1),1) a_1]*Theta1')a_3=sigmoid([ones(size(a_2,1),1) a_2]*Theta2')h_theta=a_3y=YJ_unreg=-sum(sum(y.*log(h_theta) + (1-y).*log(1-h_theta)))/m;