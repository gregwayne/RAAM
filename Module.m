classdef Module
    
    properties
        
        enc;
        dec;
        
        NH;
        NC;
        NL;
        
        init_flag;
        code_init;
        eta_init;
        
        high;
        code;
        low;
        
        mse;
        
    end
    
    methods
        
      
        function obj = Module(NL,NC,NH,NHid,g,eta_e,eta_d,eta_init)
            
            N0          = 1 + NL + NC + NH;
            N1          = NH;
            N2          = NC;
                        
            obj.enc         = SNet(N0,NHid,N2,g,eta_e);
            obj.dec         = SNet(1+NC+NH,NHid,NL+NC,g,eta_d);

            obj.NH          = NH;
            obj.NC          = NC;
            obj.NL          = NL;
            obj.eta_init    = eta_init;
            
            obj.code_init   = zeros(NC,1);
            obj.low         = zeros(NL,1);
            
            obj.mse = 0;
                        
        end
        
        function obj = SetInputs(obj,low,high)
           
            obj.low     = low;
            obj.high    = high;
            
        end
        
        function obj    = PropagateLearn(obj,niter,learn_flag,eta_rec)
            
            code_o      = LinearThreshold(obj.code);
            xIn         = [1;obj.low;code_o;obj.high]; 
            obj.enc     = obj.enc.FProp(xIn);
            code        = LinearThreshold(obj.enc.X2);
            
            if ~learn_flag
                niter = 1;
            end
            
            target          = LinearThreshold([obj.low;code_o]);
            
            p               = {};
            p.X0            = [1;code;obj.high];
            p.eta_min       = eta_rec;
            p.iters         = niter;
            p.constraint    = p.X0;
            p.cw            = 1;
            p.target        = target;
            p.cidxs         = 2:(1+obj.NC);

            obj.dec         = obj.dec.Minimize(p);
            obj.mse         = norm(target-obj.dec.X2);
            
            if learn_flag
                
                obj.dec         = obj.dec.ErrorLearn(target-obj.dec.X2,1);                
                obj.dec         = obj.dec.Tag();
                obj.dec         = obj.dec.Untag();
                                
                obj.enc         = obj.enc.FProp(xIn);                
                ch              = LinearThreshold(obj.dec.X0(2:1+obj.NC));
                obj.enc         = obj.enc.ErrorLearn(ch-obj.enc.X2,1);
                obj.enc         = obj.enc.Tag();
                obj.enc         = obj.enc.Untag();  
                
                if obj.init_flag
                    obj.code_init   = obj.code_init + obj.eta_init*(obj.dec.X2(1+obj.NL:end)-obj.code_init);
                    obj.init_flag   = 0;
                end
                                                                           
            end   
            
            obj.code    = obj.enc.X2;
            
        end
        
        function code = Encode(obj)
            
            code                = obj.enc.FastProp([1;obj.low;obj.code]);
                        
        end
        
        function [low,code_o] = Decode(obj,code,high)
           
            out                 = LinearThreshold(obj.dec.FastProp([1;code;high]));
            low                 = out(1:obj.NL);
            code_o              = out(obj.NL+1:end);
            
        end        
        
        function obj = ResetCode(obj)
           
            obj.code            = obj.code_init;
            obj.low             = 0*obj.low;
            obj.high            = 0*obj.high;
            obj.init_flag       = 1;
            
        end
                        
    end
        
end

function y = LinearThreshold(x)

    y = ((x > -1).*(x < 1)).*x - (x < -1) + (x > 1);

end