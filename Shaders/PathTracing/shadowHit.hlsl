// the closest hit of the shadow rays ( RT_HITGROUP_SHADOW , next to the any hit : the any hit lets the bodies of the lamps through , this one runs for the
// hit that was accepted ) . it has no resources : the only thing it does is to tell how far the thing that stopped the ray is
struct [raypayload] ShadowRayPayLoad{
    bool inShadow : read(closesthit , miss, caller) : write(closesthit , miss , caller);
    // the distance from the origin of the ray to what stopped it ( a hit ) , the caller sets it to the length of the ray before the trace ( nothing in the way )
    float depth : read(closesthit , miss, caller) : write(closesthit , miss , caller);
};

[shader("closesthit")]
void main(inout ShadowRayPayLoad load , in BuiltInTriangleIntersectionAttributes attr){
    load.inShadow = true;
    load.depth = RayTCurrent();
}
