
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CARTESIAN_TO_FRENET1_H_
#define ACLNN_CARTESIAN_TO_FRENET1_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCartesianToFrenet1GetWorkspaceSize
 * parameters :
 * distVec : required
 * minIdxOut : required
 * backIdxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCartesianToFrenet1GetWorkspaceSize(
    const aclTensor *distVec,
    const aclTensor *minIdxOut,
    const aclTensor *backIdxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCartesianToFrenet1
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCartesianToFrenet1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
