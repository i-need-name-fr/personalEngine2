struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

[shader("miss")]
void main(inout ShadowRayPayLoad load){
    load.inShadow = false;
}