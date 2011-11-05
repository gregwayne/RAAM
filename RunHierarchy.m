clear all;
close all;

G={};
G.fig = figure();

figure(G.fig);
clf(G.fig);

set(G.fig,'NumberTitle','off','DoubleBuffer','on',...
    'BackingStore','on','Renderer','OpenGL',...
    'Name','reverso','MenuBar','none','Position',[100, 100, 800, 800]);
axis([-1.1,1.1,-1.1,1.1]);
axis square;
axis off;

n_edges         = 100;
headx           = 0.05*cos(2*pi*(1:n_edges)/n_edges);
heady           = 0.05*sin(2*pi*(1:n_edges)/n_edges);        
G.head          = patch(headx,heady,[1 0 1]);
G.lows          = line;
set(G.lows,'LineWidth',3);
G.traj          = line;
set(G.traj,'LineWidth',3);

thetas = 0:0.3:4*pi;
PS(:,:,1) = 0.8*[sin(thetas);cos(thetas)]; 
%PS(:,:,2) = 0.6*[0.4+sin(thetas);cos(thetas)];    

tt          = size(PS,2);
ntrials     = 100;
nnets       = 5;
errors      = zeros(2,nnets,ntrials);
learn_flag  = 1;

for nnet=1:nnets
    nnet
for ttype=1:2

MakeHierarchy;

for trial=1:ntrials
    disp(trial);
    P = PS(:,:,mod(trial,size(PS,3))+1);

    for i=1:NLevels
        modules{i} = modules{i}.ResetCode();
    end
    
    for t=1:tt

        for i=1:NLevels
            
            if i==1
                lo      = P(:,t) + 0.001*randn(2,1);
            else
                lo      = [modules{i-1}.low;modules{i-1}.code];
            end
            
            if i==NLevels
                hi = 0;%modules{i-1}.code;                
            else
                hi = modules{i+1}.code;
            end
            
            modules{i}      = modules{i}.SetInputs(lo,hi);            
            modules{i}      = modules{i}.PropagateLearn(50,learn_flag,0.01);
            
        end
    end

    los         = zeros(2,tt);
    
    codes = {};
    lows  = {};

    for i=1:NLevels
        codes{i} = modules{i}.code;
        lows{i}  = modules{i}.low;
    end
    
    set(G.lows,'XData',[]);
    set(G.lows,'YData',[]);
    set(G.lows,'Color',[t/tt,0,(tt-t)/tt]);
    set(G.traj,'XData',[]);
    set(G.traj,'YData',[]);
    
    for t=tt:-1:1
        
        for i=NLevels:-1:1
           
            if i==NLevels
                hi = 0;
            else
                hi = codes{i+1};
            end
               
            [lows{i},codes{i}] = modules{i}.Decode(codes{i},hi);
            
        end

        los(:,t)    = lows{1};
        errors(ttype,nnet,trial) = errors(ttype,nnet,trial) + norm(lows{1} - P(:,t));
             
        if 1
        set(G.lows,'XData',los(1,tt:-1:t));
        set(G.lows,'YData',los(2,tt:-1:t));
        set(G.lows,'Color',[t/tt,0,(tt-t)/tt]);
        set(G.traj,'XData',P(1,1:t));
        set(G.traj,'YData',P(2,1:t));
        set(G.traj,'Color',[(tt-t)/tt,0,t/tt]);
        set(G.head,'XData',los(1,t)+headx,'YData',los(2,t)+heady);
        drawnow;
        pause(0.01);
        end

    end

end

end
end

figure(2);
plot(1:ntrials,squeeze(mean(errors(1,:,:),2)),'r',1:ntrials,squeeze(mean(errors(2,:,:),2)),'b');