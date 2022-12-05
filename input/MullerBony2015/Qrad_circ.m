function Qrad_circ 


%copied and adpated from : function MAIN_f_PNAS
%addpath /Users/Caroline/Documents/MATLAB/CMtoolbox_all/

    %irvec=9; t0=50; t1=100; nbins_target=20;  %approximate number of bins (+-1). Saved 50, 64, 120 %irvec=1 pour f01; irvec=9 pour n01
    irvec=90; t0=25; t1=30; nbins_target=15;  %approximate number of bins (+-1). Saved 50, 64, 120%90 for new 3km run done for Ben's paper, 
    %saved : 
        %t0=25; t1=30; nbins_target=15; 
        %t0=35; t1=40; nbins_target=15; 
        %t0=45; t1=50; nbins_target=15; 
        %t0=48; t1=50; nbins_target=20;
        %t0=40; t1=50; nbins_target=30; 
        t0=40; t1=50; nbins_target=64; %<- end of simulation (goes to 50 days), 30 bins.
    
    
    %%%%%%%%LOAD OR SAVE DATA 
    if 1 %load data 
        load(['Qrad_pwbinnedvariables_ir',num2str(irvec),'_t',num2str(t0),'_a',num2str(t1),'_nbins',num2str(nbins_target),'.mat'],...
              'nbins','z1d','Qrs','psis','qns','pws','rws','pws2');           
    else %save data
        
        if irvec==9
            run_name='SAM6108Outputs_n01_300_i03'; 
            fn1d='~/Desktop/Desktopp/SAMOutputs/DIR-TC-copied-nyushdrive/fplane_n01_300_i03.nc';    
            fn2d='~/Desktop/Desktopp/SAMOutputs/DIR-TC-copied-nyushdrive/fplane_n01_300_i03_256.2Dcom_1.nc';
            fn3d='~/Desktop/Desktopp/SAMOutputs/DIR-TC-copied-nyushdrive/fplane_n01_300_i03_256_0000864000.com3D.alltimes.nc';
            
        elseif irvec==90
            path='/Users/Caroline/Desktop/Desktopp/SAMOutputs/DIR-TMP/';
            fn3d=[path,'RCE_SST301_truc_256_0000432000.alltimes.nc'];
            fn2d=[path,'RCE_SST301_truc_256.2Dcom_1.nc'];
            fn1d=[path,'RCE_SST301_truc.nc'];
        end
        ncid = netcdf.open(fn3d,'nowrite');
          varid = netcdf.inqVarID(ncid,'x');    x=netcdf.getVar(ncid,varid,'double'); x3d=x'; nx=length(x); %m   [1,nx]
          varid = netcdf.inqVarID(ncid,'y');    y=netcdf.getVar(ncid,varid,'double'); y3d=y'; ny=length(y); %m   [1,ny]
          varid = netcdf.inqVarID(ncid,'z');    z=netcdf.getVar(ncid,varid,'double'); z3d=z;  nz=length(z); %m   [nz,1]
          varid = netcdf.inqVarID(ncid,'p');    p=netcdf.getVar(ncid,varid,'double');                     %hPa [nz,1]
          varid = netcdf.inqVarID(ncid,'time'); t3d=netcdf.getVar(ncid,varid,'double');
          varid = netcdf.inqVarID(ncid,'W');    wxyzt=netcdf.getVar(ncid,varid,'double'); %(x,y,z,t)
          varid = netcdf.inqVarID(ncid,'QN');   qnxyzt=netcdf.getVar(ncid,varid,'double'); %(x,y,z,t)
          varid = netcdf.inqVarID(ncid,'QRAD'); Qrxyzt=netcdf.getVar(ncid,varid,'double'); %(x,y,z,t)
        netcdf.close(ncid);    

        ncid = netcdf.open(fn2d,'nowrite');
          %varid = netcdf.inqVarID(ncid,'MSE');  msexyt=netcdf.getVar(ncid,varid,'double'); 
          varid = netcdf.inqVarID(ncid,'PW');   pwxyt=netcdf.getVar(ncid,varid,'double'); 
          varid = netcdf.inqVarID(ncid,'time'); t2d=netcdf.getVar(ncid,varid,'double');
        netcdf.close(ncid);    

        ncid = netcdf.open(fn1d,'nowrite');
          varid = netcdf.inqVarID(ncid,'RHO');  rho=netcdf.getVar(ncid,varid,'double'); 
          varid = netcdf.inqVarID(ncid,'z');    z1d=netcdf.getVar(ncid,varid,'double'); 
          varid = netcdf.inqVarID(ncid,'p');    p=netcdf.getVar(ncid,varid,'double');                     %hPa [nz,1]
          varid = netcdf.inqVarID(ncid,'time'); t1d=netcdf.getVar(ncid,varid,'double');
        netcdf.close(ncid);    

            %limit in time 
            itvec3d = [index_find(t3d,t0):1:index_find(t3d,t1)];
            ii=1;
            for it=itvec3d
                itvec2d(ii)=index_find(t2d,t3d(it)); %2d and 3d outputs do not necessarily have the same frequency (2d more frequent)
                ii=ii+1;
            end
            %rm sponge
            iz3dvec=[index_find(z3d,0):1:index_find(z3d,18e3)]; 
            iz1dvec=[index_find(z1d,0):1:index_find(z1d,18e3)]; 
            wxyzt  = squeeze(wxyzt(:,:,iz3dvec,itvec3d)); 
            qnxyzt = squeeze(qnxyzt(:,:,iz3dvec,itvec3d));
            Qrxyzt = squeeze(Qrxyzt(:,:,iz3dvec,itvec3d));
            pwxyt = squeeze(pwxyt(:,:,itvec2d));
            %time mean rho
            itvec1d = [index_find(t1d,t0):1:index_find(t1d,t1)]; rho = squeeze(rho(iz1dvec,itvec1d)); rho=squeeze(mean(rho,2));
            z1d=z1d(iz1dvec); z3d=z3d(iz3dvec); nz=length(z3d);
            if max(abs(z1d-z3d))>1e-3, error('CM: 1d and 3d z dont match pb for rho w computation'); end;
            %order by MSE
            nbins=nbins_target;
            pws=sort(pwxyt(:)); N=length(pws); 
            pws=[pws(1:floor(N/(nbins-1)):N)]; 
            nbins=length(pws)-1;
            rws=zeros(nbins,nz);qns=zeros(nbins,nz);Qrs=zeros(nbins,nz);
            psis=zeros(nbins,nz); pws2=zeros(nbins,1);
            for ibin=1:nbins
                includ=1.*(pwxyt>=pws(ibin)).*(pwxyt<pws(ibin+1));
                includ=includ(:);
                for iz=1:nz
                    w_=squeeze(wxyzt(:,:,iz,:)); w_=w_(:); rws(ibin,iz)=rho(iz)*sum(w_.*includ);
                    qn_=squeeze(qnxyzt(:,:,iz,:)); qn_=qn_(:); qns(ibin,iz)=sum(qn_.*includ)./sum(includ);
                    Qr_=squeeze(Qrxyzt(:,:,iz,:)); Qr_=Qr_(:); Qrs(ibin,iz)=sum(Qr_.*includ)./sum(includ);   
                    psis(ibin,iz)=psis(max(1,ibin-1),iz)+rws(ibin,iz);
                end
                pw_=squeeze(pwxyt(:,:,:)); pw_=pw_(:); pws2(ibin)=sum(pw_.*includ)./sum(includ);  
            end   
            %NOte that psis depends on nt here, should really divide by
            %length(t) 
            save(['Qrad_pwbinnedvariables_ir',num2str(irvec),'_t',num2str(t0),'_a',num2str(t1),'_nbins',num2str(nbins_target),'.mat'],...
           'nbins','z1d','Qrs','psis','qns','pws','rws','pws2');               
    end
      
        
    %%%%%%%%PLOT DATA
    
        unitt='kg m^{-2} s^{-1}';
    
        figure;
        %x_axis=1:nbins; xtext='rank of PW (lowest to highest)';  %plot versus PW bin
        x_axis=pws(1:end-1);xtext='PW (mm)';

        pp=pcolor(x_axis,z1d/1e3,Qrs.'); set(pp,'edgecolor','none'); colorbar; 
        
        %caxis([-5 2]); 
        caxis([min(Qrs(:)),max(Qrs(:))]);
        hold on;        
        
        dpsi= round (max(abs(psis(:)))/20);
        
        contour((x_axis),z1d/1e3,psis.',[dpsi:dpsi:max(abs(psis(:)))],'color','k','linestyle','--','linewidth',0.5);
        %contour((1:nbins),z1d/1e3,psis.',[0 0],'color',[0.5 0.5 0.5],'linestyle','-','linewidth',0.5);
        contour((x_axis),z1d/1e3,psis.',-[dpsi:dpsi:max(abs(psis(:)))],'color','k','linestyle','-','linewidth',0.5);
        %dqn=0.006; %: use same values for both 
        dqn= round (max(qns(:))/10*1e3)/1e3; 
        contour(x_axis,z1d/1e3,qns.',[dqn:dqn:max(qns(:))],'color','w','linewidth',0.5);
        xlabel(xtext); ylabel('z (km)'); 
        title([{'Self-aggregation simulations'},{['Qr (colors), \Psi (black every ',num2str(dpsi),unitt,'), qn (white every ',num2str(dqn),'g kg^{-1}) day ',num2str(t0),' to ',num2str(t1)]}]);colorbar; 
        ylim([0 15]); %text(59,10,'Qr (K day^{-1})','rotation',-90)
        set(gca,'fontsize',8,'tickdir','out','units','inches');   
        %colormap('jet');
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 






end %end function





function ii=index_find(vec, value)
    [mm,ii]=min(abs(vec-value));
end

    
