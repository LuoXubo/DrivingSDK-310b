
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CALC_POLY_START_END_SL_H_
#define ACLNN_CALC_POLY_START_END_SL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCalcPolyStartEndSlGetWorkspaceSize
 * parameters :
 * minIdx : required
 * polyLine : required
 * points : required
 * sCum : required
 * polyStartOut : required
 * polyEndOut : required
 * slOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCalcPolyStartEndSlGetWorkspaceSize(
    const aclTensor *minIdx,
    const aclTensor *polyLine,
    const aclTensor *points,
    const aclTensor *sCum,
    const aclTensor *polyStartOut,
    const aclTensor *polyEndOut,
    const aclTensor *slOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCalcPolyStartEndSl
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCalcPolyStartEndSl(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
