    M=load('heat_data_test.txt');
    dt=0.01;
    dx=0.01;
    total_time=5;
    length=5;
    t=0:dt:total_time;
    x=0:dx:length;
    
    newcol=zeros(501,1);
    M=[M newcol];
    % CO(:,:,1) = zeros(501); % red
    % CO(:,:,2) = ones(501).*linspace(0.5,0.6,501); % green
    % CO(:,:,3) = ones(501).*linspace(0,1,501); % blue
    surf(x,t,M,'edgecolor', 'none');
    xlabel('Space');
    ylabel('Time');
    zlabel('Temperature');

    colorbar;
