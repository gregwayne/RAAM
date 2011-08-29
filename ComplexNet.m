classdef ComplexNet
    
    properties
        
        net;
        sigma_sq_in;
        sigma_sq_out;
        t;
        
        X0;
        R2;
        
    end
    
    methods
        
        function obj = ComplexNet(N0,N1,N2,g,eta)
           
            obj.net     = SimpleNet(N0,N1,N2,g,eta);
            obj.t       = 1;
            obj.sigma_sq_in    = ones(N0,1);
            obj.sigma_sq_out   = ones(N2,1);
            
        end
               
        function obj = FProp(obj,X0)
            
            % Time is update at the beginning of every FProp
            obj.t   = obj.t + 1;
            t       = obj.t;
            
            % Divide by standard deviation of inputs
            obj.X0  = X0./sqrt(obj.sigma_sq_in);
            obj.net = obj.net.FProp(X0);
            % Multiply by standard deviation of outputs
            obj.R2  = obj.net.R2.*sqrt(obj.sigma_sq_out);                        
            
            obj.sigma_sq_in     = (1/t)*X0.^2 + obj.sigma_sq_in.*(t-1)/t;
            
        end             
                                     
        function obj = ErrorLearn(obj,target)
           
            t       = obj.t;
            target  = target./sqrt(obj.sigma_sq_out);
            obj.net = obj.net.ErrorLearn(target-obj.net.R2);
            
            obj.sigma_sq_out    = (1/t)*target.^2 + obj.sigma_sq_out.*(t-1)/t; 
            
        end
        
        function obj = Tag(obj)
           
            obj.net = obj.net.Tag();
            
        end
        
        function obj = Untag(obj)
           
            obj.net = obj.net.Untag();
            
        end
        
        
    end
    
end