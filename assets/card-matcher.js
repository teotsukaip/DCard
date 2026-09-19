/* Whole-image ORB matching, validated by a RANSAC projective transform. */
(function(root){
class CardMatcher {
 constructor(cv, rgba){
  this.cv=cv;this.orb=new cv.ORB();this.orb.setMaxFeatures(1400);this.orb.setEdgeThreshold(19);this.orb.setFastThreshold(12);this.matcher=new cv.BFMatcher(cv.NORM_HAMMING,false);
  this.keypoints=new cv.KeyPointVector();this.descriptors=new cv.Mat();this.width=rgba.cols;this.height=rgba.rows;
  const gray=new cv.Mat(),mask=new cv.Mat();
  try{cv.cvtColor(rgba,gray,cv.COLOR_RGBA2GRAY);this.orb.detectAndCompute(gray,mask,this.keypoints,this.descriptors);if(this.descriptors.rows<20)throw Error('Reference image has too few features')}finally{gray.delete();mask.delete()}
 }
 match(rgba){
  const cv=this.cv,owned=[];const own=v=>(owned.push(v),v);
  try{
   const gray=own(new cv.Mat()),kp=own(new cv.KeyPointVector()),desc=own(new cv.Mat()),mask=own(new cv.Mat());
   cv.cvtColor(rgba,gray,cv.COLOR_RGBA2GRAY);this.orb.detectAndCompute(gray,mask,kp,desc);
   if(desc.rows<20)return {matched:false,inliers:0};
   const matches=own(new cv.DMatchVectorVector());this.matcher.knnMatch(this.descriptors,desc,matches,2);
   const pairs=[],used=new Set();
   for(let i=0;i<matches.size();i++){const v=matches.get(i);try{if(v.size()<2)continue;const a=v.get(0),b=v.get(1);if(a.distance<58&&a.distance<.72*b.distance&&!used.has(a.trainIdx)){pairs.push([this.keypoints.get(a.queryIdx).pt,kp.get(a.trainIdx).pt]);used.add(a.trainIdx)}}finally{v.delete()}}
   if(pairs.length<14)return {matched:false,inliers:0};
   const a=own(cv.matFromArray(pairs.length,1,cv.CV_32FC2,pairs.flatMap(p=>[p[0].x,p[0].y]))),b=own(cv.matFromArray(pairs.length,1,cv.CV_32FC2,pairs.flatMap(p=>[p[1].x,p[1].y]))),inlierMask=own(new cv.Mat());
   const h=own(cv.findHomography(a,b,cv.RANSAC,4,inlierMask));if(h.empty())return {matched:false,inliers:0};
   const good=pairs.filter((_,i)=>inlierMask.data[i]);const inliers=good.length;
   if(inliers<14||inliers/pairs.length<.5)return {matched:false,inliers};
   const xs=good.map(p=>p[0].x),ys=good.map(p=>p[0].y),coverage=(Math.max(...xs)-Math.min(...xs))*(Math.max(...ys)-Math.min(...ys))/(this.width*this.height);
   const corners=own(cv.matFromArray(4,1,cv.CV_32FC2,[0,0,this.width,0,this.width,this.height,0,this.height])),out=own(new cv.Mat());cv.perspectiveTransform(corners,out,h);
   const pts=Array.from({length:4},(_,i)=>({x:out.data32F[2*i],y:out.data32F[2*i+1]}));
   if(pts.some(p=>!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x< -rgba.cols*.15||p.x>rgba.cols*1.15||p.y< -rgba.rows*.15||p.y>rgba.rows*1.15))return {matched:false,inliers};
   const cross=pts.map((p,i)=>{const q=pts[(i+1)%4],r=pts[(i+2)%4];return (q.x-p.x)*(r.y-q.y)-(q.y-p.y)*(r.x-q.x)});
   const convex=cross.every(n=>n>0)||cross.every(n=>n<0);const area=Math.abs(pts.reduce((s,p,i)=>{const q=pts[(i+1)%4];return s+p.x*q.y-p.y*q.x},0))/2;
   return {matched:convex&&coverage>.12&&area>rgba.cols*rgba.rows*.045&&area<rgba.cols*rgba.rows*1.2,inliers,coverage};
  }finally{owned.reverse().forEach(v=>v.delete())}
 }
 delete(){this.orb.delete();this.matcher.delete();this.keypoints.delete();this.descriptors.delete()}
}
if(typeof module!=='undefined'&&module.exports)module.exports=CardMatcher;else root.CardMatcher=CardMatcher;
})(typeof window==='undefined'?globalThis:window);
