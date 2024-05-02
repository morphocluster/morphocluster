import axios from "axios";

export function getNode(node_id) {
    return axios.get(`/api/nodes/${node_id}`)
        .then(response => {
            return response.data;
        });
}

export function patchNode(node_id, data) {
    return axios.patch(`/api/nodes/${node_id}`, data)
        .then(response => {
            return response.data;
        });
}

/**
 * Get the next unapproved node for a subtree rooted at node_id.
 *
 * @param params {leaf: bool}
 */
export function getNextUnapprovedNode(node_id, params = null) {
    return axios.get(`/api/nodes/${node_id}/next`, { params })
        .then(response => {
            return response.data;
        });
}

/**
 * Get the next unfilled node for a subtree rooted at node_id.
 *
 * @param params {leaf: bool, preferred_first: bool, order_by: string}
 */
export function getNextUnfilledNode(node_id, params = null) {
    return axios.get(`/api/nodes/${node_id}/next_unfilled`, { params })
        .then(response => {
            return response.data;
        });
}

/**
 * Get the sorting progress for a subtree rooted at node_id.
 *
 * @param params {log: string}
 */
export function getNodeProgress(node_id, params = null) {
    return axios.get(`/api/nodes/${node_id}/progress`, { params })
        .then(response => {
            return response.data;
        });
}

/**
 * Get recommended objects for node_id.
 *
 * @param params {max_n: int}
 */
export function getNodeRecommendedObjects(node_id, params = null) {
    return axios.get(`/api/nodes/${node_id}/recommended_objects`, { params })
        .then(response => {
            return response.data;
        });
}

export function mergeNodeInto(node_id, dest_node_id) {
    const data = { dest_node_id };
    console.log(data)
    return axios.post(`/api/nodes/${node_id}/merge_into`, data);
}

// Files
// TODO: Documentation
export function getDirEntry(file_path) {
    return axios.get(`/api/files/${file_path}?download=false&info=true`)
        .then(response => {
            return response.data;
        });
}

export function getFileInfo(file_path) {
    return axios.get(`/api/files/${file_path}?download=false&info=true`)
        .then(response => {
            return response.data;
        });
}

export function getFile(file_path) {
    return axios.get(`/api/files/${file_path}?download=true&info=false`)
        .then(response => {
            return response.data;
        });
}

export function uploadFiles(files, file_path) {
    return axios.post(`/api/files/${file_path}`, files)
        .then(response => { return response.data; });
}


// Project

export function getProjects(include_progress = false) {
    return axios.get(`/api/projects`, { params: { include_progress } })
        .then(response => {
            return response.data;
        });
}

export function getProject(project_id, include_progress = false) {
    return axios.get(`/api/projects/${project_id}`, { params: { include_progress } })
        .then(response => {
            return response.data;
        });
}

export function saveProject(project_id) {
    return axios.post(`/api/projects/${project_id}/save`)
        .then(response => {
            return response.data;
        });
}

export function nodeAdoptMembers(node_id, members) {
    if (!Array.isArray(members)) {
        members = [members];
    }

    return axios.post(`/api/nodes/${node_id}/adopt_members`, { members: members });
}

export function nodeAcceptRecommendations(node_id, request_id, rejected_members, last_page, log_data = null) {
    return axios.post(`/api/nodes/${node_id}/accept_recommended_objects`,
        { request_id, rejected_members, last_page, log_data });
}

export function getUnfilledNodes(project_id) {
    return axios.get(`/api/projects/${project_id}/unfilled_nodes`).then(response => {
        return response.data;
    });
}

export function log(action, node_id = null, reverse_action = null, data = null) {
    return axios.post(`/api/log`,
        { action, node_id, reverse_action, data });
}