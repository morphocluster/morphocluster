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

export function getNextUnapprovedNode(node_id, leaf = false) {
    return axios.get(`/api/nodes/${node_id}/next`, { params: { leaf } })
        .then(response => {
            return response.data;
        });
}

export function getNextUnfilledNode(node_id, leaf = false) {
    return axios.get(`/api/nodes/${node_id}/next_unfilled`, { params: { leaf } })
        .then(response => {
            return response.data;
        });
}

export function getNodeProgress(node_id, log = null) {
    var params = {}
    if (log !== null) {
        params.log = log;
    }
    return axios.get(`/api/nodes/${node_id}/progress`, { params })
        .then(response => {
            return response.data;
        });
}

export function getNodeRecommendedObjects(node_id, max_n = null) {
    var params = {}
    if (max_n !== null) {
        params.max_n = max_n;
    }
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