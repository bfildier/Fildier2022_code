function [Tma,Tpotma,pma,qvma,zgrd]=bcompute_moist_adiabatic_profile
%%%% 2021 : adapt from 2017-Sara/compute_moist_adiabatic_profile
%%%% 2016: computed idealized moist adiabatic profiles to compare with LMDZ
%idealized moist adiabat (simplified version from earlier codes since
%here I don't know p a priori. So I simplified the computation).


%%%% SST VECTOR:
    SST_vec=296:2:340;

%%%% PARAMETERS:
      RHsfc=80/100;        %relative humidity near the surface (used to compute the LCL)
      psfc=1010;           %surface pressure in hPa
      zgrd=[0:50:18000].'; %altitude in m

%%%% INITIALIZE VARIABLES
    Tma=zeros(length(zgrd),length(SST_vec)); %moist adiabatic temperature in K
    pma=Tma; %moist adiabatic pressure in hPa
    qvma=Tma; %moist adiabatic water vapor mixing ratio in g/kg
    prespot=Tma;
    Tpotma=Tma;
    izLCL=zeros(1,length(SST_vec));
    for iSST=1:length(SST_vec)
        SST=SST_vec(iSST);
          Tsfc=SST-1;          %jump in temperature between ocean and air    
          %%%% COMPUTE MOIST ADIABATIC ASCENT
            [Tma(:,iSST),pma(:,iSST),qvma(:,iSST),izLCL(iSST)]=parcel_ascent_SAM6103_simple(zgrd,psfc,Tsfc,RHsfc); 
             prespot(:,iSST)=(1000./pma(:,iSST)).^(bPARS_SAM_params('rgas')/bPARS_SAM_params('cp')); Tpotma(:,iSST)=Tma(:,iSST).*prespot(:,iSST); 
             write_data_in_file(['DATA_MOIST_ADIABAT_SST_',num2str(SST)], ['Moist adiabatic profiles SST=',num2str(SST)], zgrd,Tma(:,iSST),qvma(:,iSST));  
    end
    
%%%% Plot to check if makes sense: Compare with some past sounding
    figure; 
        cmap=colormap('jet');
        dcmap=floor(size(cmap,1)/length(SST_vec));
        cmap=cmap(1:dcmap:end,:);
        for iSST=1:length(SST_vec)
            SST=SST_vec(iSST);
            subplot(221); hold on; plot(Tma(:,iSST), zgrd/1e3,'color',cmap(iSST,:)); 
                if SST==SST_vec(end), title([{['Idealized moist adiabatic profiles ']},{['T-ma']}]); ylabel('z km'); end
            subplot(222);hold on; plot(Tpotma(:,iSST), zgrd/1e3,'color',cmap(iSST,:)); 
                         %hold on; plot(Tpotma(:,iSST).*exp(bPARS_SAM_params('Lv')*qvma(:,iSST)./(Tma(:,iSST)*bPARS_SAM_params('cp'))), zgrd/1e3,'color',cmap(iSST,:),'linestyle','--'); 
                         %hold on; plot( (Tma(:,iSST)*bPARS_SAM_params('cp')+bPARS_SAM_params('grav')*zgrd+ bPARS_SAM_params('Lv')*qvma(:,iSST))/bPARS_SAM_params('cp'),zgrd/1e3,'color',cmap(iSST,:),'linestyle','--'); 
                if SST==SST_vec(end), title('\theta-ma');  legend(num2str(SST_vec.'));  end 
            subplot(223); hold on; plot(pma(:,iSST), zgrd/1e3,'color',cmap(iSST,:)); 
                if SST==SST_vec(end), title('p-ma');end
            subplot(224);hold on; plot(qvma(:,iSST)*1e3, zgrd/1e3,'color',cmap(iSST,:)); 
                if SST==SST_vec(end), title('qv-ma g/kg'); end;
                line([min(qvma(:,iSST)*1e3),max(qvma(:,iSST)*1e3)],[1 1]*zgrd(izLCL(iSST))/1e3,'color',cmap(iSST,:),'linestyle','--'); 
                legend('qv g/kg','LCL');
        end
end       







function [Tma,pma,qvma,izLCL]=parcel_ascent_SAM6103_simple(zgrd,psfc,Tsfc,RHsfc)
    %Tma: temp[K] of a parcel raising adiabatically from the 
    %       surface Tsurface=Tin(1) to its LCL on a dry adiab and then
    %       following a moist adiab
    %qvma(kg/kg): constant to LCL then saturated;
    %pma (hPa).
    %(edited starting from MATLAB/TropHeight/compute_snd_file.m needed to
    %change since here I don't have p(z) or T(z)=> I simplified the computation here) 
    psfcPa=psfc*100; %Pa
    
    %SAM params.f90
        cp    = bPARS_SAM_params('cp');  %J/kg/K
        ggr   = bPARS_SAM_params('grav'); %m/s2
        lcond = bPARS_SAM_params('Lv');   %! Latent heat of condensation, J/kg
        lfus  = bPARS_SAM_params('Lf');   %! Latent heat of fusion, J/kg
        lsub  = bPARS_SAM_params('Ls');   %! Latent heat of sublimation, J/kg
        Rv    = bPARS_SAM_params('rv');          %! Gas constant for water vapor, J/kg/K
        Rd    = bPARS_SAM_params('rgas');          %! Gas constant for dry air, J/kg/K
        eps   = Rd/Rv;
        epsi  = 0.61; %should be 1/eps-1; but take value used in SAM
        
    %initialize
        nz   = length(zgrd);
        Tma  = zeros(nz,1);
        pma  = zeros(nz,1);
        qvma = zeros(nz,1);
        lma  = zeros(nz,1);
        
        
    %1) 1st level to LCL (1st saturated level).
    qssfc=saturation_mixing_ratio_SAM6103(Tsfc,psfcPa);
    qvsfc=RHsfc*qssfc;
    [zLCL,izLCL,Tz,pz,qsatz] = find_LCL_simple(zgrd,psfcPa,Tsfc,qvsfc,cp,ggr,Rd); %find LCL zLCL:1st SATURATED level
        Tma(1:(izLCL-1))   = Tz(1:(izLCL-1));  
        pma(1:(izLCL-1))   = pz(1:(izLCL-1));  
        qvma(1:(izLCL-1))  = qvsfc;
        lma(1:(izLCL-1))   = 0; %condensate amount
        
        
        
        
                
    %2) LCL to top: moist adiab (in SAM, cpT + gz - Ls qs -Lv ql = cstt, and qv+qs+ql=cstt and qv=qsat).
    for iz=izLCL:nz
        t0=Tma(iz-1); p0=pma(iz-1); z0=zgrd(iz-1); qv0=qvma(iz-1);l0=lma(iz-1);
        z1=zgrd(iz); p1=p0*exp(-ggr*(z1-z0)/Rd/t0);

        [t1,qv1,l1]=solve_for_temp_higher(t0,p0,z0,qv0,l0,z1,p1,cp,ggr,lcond,lsub);
        qvma(iz)=qv1; lma(iz)=l1; Tma(iz)=t1; pma(iz)=p1;
    end


    pma=pma/100; %hPa
            
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rs=saturation_mixing_ratio_SAM6103(t,pPa)
    p=pPa/100; %p in mbar; rs in kg/kg, t in K
    %from SAM code sat.f90 (originally from Flatau et al 92 "polynomial fits to saturation vapor pressure")
    %then r=(Rv/Rd)*e/(p-e) (not sure why there is a max in SAM's formulas...)
    
    %real function esatw(t)
    aw=[6.11239921, 0.443987641, 0.142986287e-1, 0.264847430e-3, 0.302950461e-5, 0.206739458e-7, 0.640689451e-10, -0.952447341e-13,-0.976195544e-15];
    aw0=aw(1); aw1=aw(2); aw2=aw(3); aw3=aw(4); aw4=aw(5); aw5=aw(6); aw6=aw(7); aw7=aw(8); aw8=aw(9); 
    dt = max(-80.,t-273.16);
    esatw = aw0 + dt.*(aw1+dt.*(aw2+dt.*(aw3+dt.*(aw4+dt.*(aw5+dt.*(aw6+dt.*(aw7+aw8.*dt)))))));

    %real function qsatw(t,p) !p(mb)
    esat = esatw;
    qsatw = 0.622 * esat./max(esat,p-esat);

    %real function esati(t)
    ai=[6.11147274, 0.503160820, 0.188439774e-1,0.420895665e-3, 0.615021634e-5,0.602588177e-7,0.385852041e-9, 0.146898966e-11, 0.252751365e-14]; 
    ai0=ai(1); ai1=ai(2); ai2=ai(3); ai3=ai(4); ai4=ai(5); ai5=ai(6); ai6=ai(7); ai7=ai(8); ai8=ai(9);  
    
    dt = (t>185).*(t-273.16) + (t<=185).*max(-100.,t-273.16);
    esati= (t>273.15).*esatw + ...
           (t<=273.15).*(t>185).*(ai0 + dt.*(ai1+dt.*(ai2+dt.*(ai3+dt.*(ai4+dt.*(ai5+dt.*(ai6+dt.*(ai7+ai8.*dt)))))))) + ...
           (t<=185).*(0.00763685 + dt.*(0.000151069+dt*7.48215e-7));


    %real function qsati(t,p) !p(mb)    
    esat=esati;
    qsati=0.622 * esat./max(esat,p-esat);
    
    om=omega_n(t);
    rs=om.*qsatw+(1-om).*qsati;

end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [zLCL,izLCL,Tz,pz,qsatz] = find_LCL_simple(zgrd,p0Pa,T0,qv0,cp,ggr,Rd)
    %Find zLCL s.t. qsat(T(z),p(z))=qv0 where T(z)=t0-ggr/cp*z and dp/p=-gdz/RT
    Tz      = T0-ggr/cp*zgrd;
    pz(1)   = p0Pa;
    for iz=2:length(zgrd)
        ddz=zgrd(iz)-zgrd(iz-1);
        pz(iz)=pz(iz-1)*exp(-ggr*ddz/Rd/Tz(iz-1));
    end
    pz=pz.';
    qsatz=saturation_mixing_ratio_SAM6103(Tz,pz);
    iz      = find(qsatz>qv0,1,'last'); 
    izLCL   = iz+1; %I am assuming that izLCL<=nz
    zLCL    = zgrd(izLCL); 
    
end



        
function [t1,r1,l1]=solve_for_temp_higher(t0,p0,z0,r0,l0,z1,p1,cp,ggr,lcond,lsub)
    %solve t0 + g/cp*z0 - l0/cp*( lcond*om(t0) + lsub*(1-om(t0)) ) = t1 + g/cp*z1 - l1(t1)/cp*( lcond*om(t1) + lsub*(1-om(t1)) ) 
    %where l1(t1)=l0+(r0-rs(t1,p1))
    %use newton Raphson (as in SAM clouds.f90)

    %initial guess for temperature    
    dr=r0-saturation_mixing_ratio_SAM6103(t0+ggr/cp*(z0-z1), p1);
    l1=l0+dr;
    r1=r0-dr;    
    t1=t0 + ggr/cp*(z0-z1) - (l0-l1)/cp*( (lcond-lsub)*omega_n(t0) + lsub); 
    dr=r0-saturation_mixing_ratio_SAM6103(t1, p1);
    l1=l0+dr;
    r1=r0-dr;        
    
    %%%disp('*** -> in solve_for_temp_higher subroutine. Starting iteration');
    %%%disp(['t0=',num2str(t0),'  r0=',num2str(r0),'  l0=',num2str(l0),'  z0=',num2str(z0),'  z1=',num2str(z1),'  rsat(t0,p0)=',num2str(saturation_mixing_ratio_SAM(t0,p0)),' (should be saturated at z0)']);
    %%%disp(['t1=',num2str(t1),'  t0+g/cp*(z0-z1)-l0/cp*( (Lc-Ls)*om(t0)+Ls )+l1/cp*( (Lc-Ls)*om(t1)+Ls )=',num2str(t0+ggr/cp*(z0-z1)-l0/cp*( (lcond-lsub)*omega_n(t0)+lsub )+l1/cp*( (lcond-lsub)*omega_n(t1) + lsub))]);
    niter=0;
    dtabs = 100;    
    while (abs(dtabs) > 0.01)&(niter<10)
        ff=f_higher(t1,t0,p0,z0,r0,l0,z1,p1,cp,ggr,lcond,lsub);
        dff=(ff-f_higher(t1-.002,t0,p0,z0,r0,l0,z1,p1,cp,ggr,lcond,lsub) )/.002;
        dtabs=-ff/dff;
        niter=niter+1;
        t1=t1+dtabs;
        dr=r0-saturation_mixing_ratio_SAM6103(t1, p1);
        l1=l0+dr;
        r1=r0-dr;                
        %%%disp(['iter=',num2str(niter),' r0=',num2str(r0),' r1=',num2str(r1),' l0=',num2str(l0),' l1=',num2str(l1),' t0=',num2str(t0),' t1=',num2str(t1),...
        %%%    '  t0+g/cp*(z0-z1)-l0/cp*( (Lc-Ls)*om(t0)+Ls )+l1/cp*( (Lc-Ls)*om(t1)+Ls )=',num2str(t0+ggr/cp*(z0-z1)-l0/cp*( (lcond-lsub)*omega_n(t0)+lsub )+l1/cp*( (lcond-lsub)*omega_n(t1) + lsub))])
    end                  
    %%%disp(['hit a key to continue (t1=',num2str(t1),'  t0+g/cp*(z0-z1)-l0/cp*( (Lc-Ls)*om(t0)+Ls )+l1/cp*( (Lc-Ls)*om(t1)+Ls )=',num2str(t0+ggr/cp*(z0-z1)-l0/cp*( (lcond-lsub)*omega_n(t0)+lsub )+l1/cp*( (lcond-lsub)*omega_n(t1) + lsub))]); %pause; 
    %%%disp(['*** <- solve_for_temp_higher (rsat(t1,p1)=',num2str(saturation_mixing_ratio_SAM(t1,p1))]);
end


function ff=f_higher(t,t0,p0,z0,r0,l0,z1,p1,cp,ggr,lcond,lsub)
    % want T such that T+g/cp*z1-l1(T)/cp*(lcond*om(T) + lsub*(1-om(T))) = T0+g/cp*z0-l0/cp*(lcond*om(T0) + lsub*(1-om(T0))) 
    % where l1(T)=l0 + dr(T), dr(T)=r0-rsat(T,p)
    dr=r0-saturation_mixing_ratio_SAM6103(t, p1);
    l1=l0+dr;  
    ff=t-( t0+ggr/cp*(z0-z1)-l0/cp*( (lcond-lsub)*omega_n(t0)+lsub )+l1/cp*( (lcond-lsub)*omega_n(t) + lsub) );
end

function om=omega_n(t)
    %see cloud.f90 in MICRO DIR
    t00n=253.16;
    t0n =273.16;
    om=max(0,min(1,(t-t00n)/(t0n-t00n)));    
end
        




function write_data_in_file(file_name, file_info, zma,Tma,qvma)  
  %write data T(z), qv(z) etc to file_name
 disp(' '); disp(['*****: WRITING ',file_name]); 
 %format_data=['% 10.1f % d \n']; 
 format_data=['%10.0f %10.3f %10.10f  \n']; 
 nlin=length(zma);
 if size(Tma)~=[nlin,1], error(['size of Tma =[',num2str(size(Tma,1)),',',num2str(size(Tma,2)),'] but [nlin,1]=',num2str(nlin),',1']); end;
 if size(qvma)~=[nlin,1], error(['size of Tma =[',num2str(size(Tma,1)),',',num2str(size(Tma,2)),'] but [nlin,1]=',num2str(nlin),',1']); end;
 data3=[zma  Tma  qvma]
 
     fid = fopen(file_name,'wt'); 
     fprintf(fid,'%s \n',[file_info]);
     fprintf(fid,'%s \n',[' z(m)  T(K)  qv(kg/kg)']); 

     for iline=1:nlin, fprintf(fid,format_data,data3(iline,:)); end
     
         fclose(fid); 
         disp(' '); 
end



