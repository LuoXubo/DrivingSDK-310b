
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GATHER_NMS3D_MASK_H_
#define ACLNN_GATHER_NMS3D_MASK_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGatherNms3dMaskGetWorkspaceSize
 * parameters :
 * mask : required
 * keepOut : required
 * numOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGatherNms3dMaskGetWorkspaceSize(
    const aclTensor *mask,
    const aclTensor *keepOut,
    const aclTensor *numOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGatherNms3dMask
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGatherNms3dMask(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
