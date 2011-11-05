classdef SNet
    
    properties

        W1; % First Layer Weight
        W2; 
        
        R0;
        X0;
        X1;
        R1;
        X2;
        R2;
        D0;
        D1;
        D2;
        
        N0;
        N1;
        N2;
        
        t;
        F;
        Fr;
        eta;
        
        elig;

    end
    
    methods
        
        function obj = SNet(N0,N1,N2,g,eta)
            
            obj.N0      = N0;
            obj.N1      = N1;
            obj.N2      = N2;
            
            obj.W1      = g*randn(N1,N0)/sqrt(N0);
            obj.W2      = 0*g*randn(N2,N1)/sqrt(N1);
            
            obj.X1      = zeros(N1,1);
            obj.R1      = zeros(N1,1);
            obj.X2      = zeros(N2,1);
            obj.R2      = zeros(N2,1);
            
            obj.t       = 2;
            obj.F       = eye(N1,N1);
            obj.Fr      = zeros(N1,1);
            obj.eta     = eta;
            
            obj.elig    = zeros(N2,N1);
            
        end
        
        function obj = FProp(obj,X0)
            
            obj.X0      = X0;
            obj.R0      = tanh(obj.X0);
            
            obj.X1      = obj.W1*obj.R0;
            obj.R1      = tanh(obj.X1);
            
            obj.X2      = obj.W2*obj.R1;
            obj.R2      = obj.X2; % Option for a Transfer Function
            
        end
        
        function obj = BProp(obj,E2)
                       
            obj.D2  = E2;
            E1      = obj.W2'*obj.D2;
            obj.D1  = (1-obj.R1.^2).*E1;
            E0      = obj.W1'*obj.D1;
            obj.D0  = (1-obj.R0.^2).*E0;
            
        end
        
        function obj = Minimize(obj,p)
            
            X0              = p.X0;
            eta_min         = p.eta_min;
            iters           = p.iters;
            constraint      = p.constraint;
            cw              = p.cw;
            target          = p.target;
            cidxs           = p.cidxs;
            
            for iter=1:iters
                       
                obj         = obj.FProp(X0);
                obj         = obj.BProp(target-obj.R2);
                sc          = cw*(constraint-X0); % soft constraint
                X0(cidxs)   = LinearThreshold(X0(cidxs) + eta_min*obj.D0(cidxs) + eta_min*sc(cidxs));

            end
                       
        end
        
        function X2 = FastProp(obj,X0)
           
            R0          = tanh(X0);
            X1          = obj.W1*R0;
            R1          = tanh(X1);
            X2          = obj.W2*R1;
            
        end        
                
        function obj = Fisher(obj)
           
            eps_t       = 1/obj.t;
            meps_t      = 1-eps_t;
            
            obj.t       = obj.t + 1;
            F           = obj.F;
            r           = obj.R1;
            Fr          = F*r;
            obj.Fr      = Fr;
            
            obj.F       = (meps_t)^(-1)*(F - eps_t*Fr*Fr'/(meps_t + eps_t*r'*Fr));            
            obj.t       = obj.t + 1;

        end       
                
        function obj = ErrorLearn(obj,delta,fisher_on)
           
            if fisher_on
                obj         = obj.Fisher();
            end
            obj.elig    = obj.elig + obj.eta*delta*obj.Fr'/(obj.R1'*obj.Fr); 
            
        end
        
        function obj = Tag(obj)
            
            obj.W2      = obj.W2 + obj.elig;
            
        end
        
        function obj = Untag(obj)
           
            obj.elig    = 0*obj.elig;
            
        end
        
    end
    
end

function y = LinearThreshold(x)

    y = ((x > -1).*(x < 1)).*x - (x < -1) + (x > 1);

end