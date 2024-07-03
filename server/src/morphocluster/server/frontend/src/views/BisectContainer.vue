<template>
    <div id="bisect-container">
        <nav class="navbar navbar-expand-lg navbar-light bg-dark">
            <router-link class="navbar-brand text-light" to="/">MorphoCluster</router-link>
            <div class="collapse navbar-collapse">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item nav-link text-light" v-if="project">
                        {{project.name}}
                    </li>
                    <li class="nav-item nav-link  text-light">
                        Bisect
                    </li>
                    <!-- <li class="nav-item  nav-link text-light" v-if="current_data">
                        {{current_data.node.name}}
                    </li> -->
                </ul>
            </div>
        </nav>
        <!-- The core component -->
        <DummyComponent class="core-component" v-if="current_data" :key="current_key" :data="current_data" />
        <div class="spinner-container" v-else>
            <spinner></spinner>
        </div>
        <b-modal ref="doneModal" lazy centered no-fade header-bg-variant="success" title="Growing done">
            <div class="d-block text-center">
                Growing is done for this project.
            </div>
            <footer slot="modal-footer">
                <b-button variant="primary" :to="{name: 'projects'}">Back to projects</b-button>
            </footer>
        </b-modal>
    </div>
</template>

<script>
import axios from "axios";
import shuffle from "lodash/shuffle";
import Spinner from "vue-simple-spinner";

import mixins from "@/mixins.js";

import * as api from "@/helpers/api.js";

import MessageLog from "@/components/MessageLog.vue";
import DummyComponent from "@/components/DummyComponent.vue";

const MAX_N_RECOMMENDATIONS = 100000;

export default {
    name: "bisect",
    data() {
        return {
            project: null,
            project_loading_queue: null,
            project_wp_promises: [],

            current_key: 0,
            current_data: null
        };
    },
    components: {
        MessageLog,
        Spinner,
        DummyComponent
    },
    mixins: [mixins],
    watch: {
        $route: "initialize"
    },
    created() {
        this.initialize();
    },
    methods: {
        updateCore(data) {
            this.current_data = data;
            this.current_key++;
        },
        initialize() {
            console.log(this.$route.params);

            var p;

            p = new Promise(resolve => {
                if (
                    this.project &&
                    this.project.project_id == this.$route.params.project_id
                ) {
                    // Project was already loaded.
                    resolve();
                } else {
                    // Load a project and its queue
                    this.loadProject(this.$route.params.project_id).then(() => {
                        resolve();
                    });
                }
            });

            p = p.then(() => {
                console.log("Project loaded.");
            });

            p = p.then(() => {
                if (this.project_loading_queue.length == 0) {
                    this.$refs.doneModal.show();
                    throw null;
                }
            });

            // As soon as the project with its queue is loaded, begin loading a working package in parallel to the following.
            p.then(() => {
                this.loadWorkingPackage();
            });

            p = p.then(() => {
                // If the route already contains a node_id, return it
                if (this.$route.params.node_id) {
                    return parseInt(this.$route.params.node_id);
                }

                // Otherwise use the id that is currently loading
                if (!Object.keys(this.project_wp_promises).length) {
                    throw new Error("No working package promises.");
                }

                var node_id = Object.keys(this.project_wp_promises).pop();

                const to = {
                    name: "bisect",
                    params: {
                        project_id: this.project.project_id,
                        node_id: node_id
                    }
                };

                this.$router.replace(to);

                // Throw to prevent double execution of the following
                throw null;
            });

            p = p.then(node_id => {
                console.log("node_id is now known:", node_id);
                return node_id;
            });

            //TODO: Go on.
            // Now we need to get the working package promise with the node_id from the queue and provide the component with the correct data after resolution.

            p.catch(err => {
                if (err === null) {
                    console.log("No error.");
                } else {
                    console.log(err);
                }
            });
        },
        /**
         * Load a project and a list of unfilled nodes.
         */
        loadProject(project_id) {
            this.project = null;

            var projectPromise = api
                .getProject(project_id, true)
                .then(project => {
                    this.project = project;
                });

            var queuePromise = api
                .getUnfilledNodes(project_id)
                .then(unfilled_nodes => {
                    this.project_loading_queue = unfilled_nodes;
                });
            return Promise.all([projectPromise, queuePromise]);
        },
        loadWorkingPackage(node_id = null) {
            if (node_id === null) {
                node_id = this.project_loading_queue.pop();
            } else {
                // TODO: Remove node_id from project_loading_queue
            }
            console.log(`Loading working package for ${node_id}...`);

            // Load the node
            var nodePromise = api.getNode(node_id);

            // Get the URL for the cached members
            var membersUrl = `/api/nodes/${node_id}/members?objects=1&nodes=0&arrange_by=random&page=0`;
            var membersUrlPromise = axios.get(membersUrl).then(response => {
                return response.data.links.self;
            });

            // Get the URL for the cached recommendations
            var recommendationsUrlPromise = api
                .getNodeRecommendedObjects(node_id, {max_n: MAX_N_RECOMMENDATIONS})
                .then(data => {
                    return data.links.self;
                });

            var allPromise = Promise.all([
                nodePromise,
                membersUrlPromise,
                recommendationsUrlPromise
            ]);

            // Put promise into collection
            this.project_wp_promises[node_id] = allPromise;

            allPromise.then((node, membersUrl, recommendationsUrl) => {
                console.log(`Working package for ${node_id} is now loaded.`);
                console.log(node, membersUrl, recommendationsUrl);
            });

            return allPromise;
        },

        membersOk: function() {
            this.rec_interval_left = this.rec_current_page + 1;

            if (!this.found_right) {
                this.rec_current_page = Math.min(
                    this.rec_current_page + this.jump_pages,
                    this.rec_n_pages - 1
                );
                this.jump_pages *= 2;
            } else {
                this.updateCurrentPage();
            }

            this.showNext();
        },
        membersNotOk: function() {
            this.rec_interval_right = this.rec_current_page;
            this.found_right = true;

            this.updateCurrentPage(0.25);

            this.showNext();
        },
        updateCurrentPage(frac = 0.5) {
            this.rec_current_page = Math.trunc(
                (1 - frac) * this.rec_interval_left +
                    frac * this.rec_interval_right
            );
        },
        showNext: function() {
            console.log(
                this.rec_current_page,
                this.rec_interval_left,
                this.rec_interval_right
            );

            if (this.n_unsure_pages <= 0) {
                this.done = true;

                this.saveResult();
                return;
            }

            axios
                .get(`${this.rec_base_url}&page=${this.rec_current_page}`)
                .then(response => {
                    console.log(
                        response.data.data,
                        shuffle(response.data.data)
                    );
                    this.rec_members = shuffle(response.data.data);
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        saveResult() {
            // For each page in the valid range (0<=x<this.rec_interval_left):
            // Fetch members and assign to the current node.

            console.log("Saving...");
            this.saving = true;
            this.saving_start_ms = Date.now();

            // Save all data of the current run.
            // If the user continues with the next node, all data is lost.
            var node = this.node;

            api.nodeAcceptRecommendations(
                node.node_id,
                this.rec_request_id,
                this.rejected_members,
                this.rec_interval_left
            )
                .then(() => {
                    console.log("Saved all recommendations.");
                })
                .then(() => {
                    return api.patchNode(node.node_id, { filled: true });
                })
                .then(() => {
                    console.log("Saved.");
                    this.saving = false;
                    this.saved = true;
                    this.saving_total_ms = Date.now() - this.saving_start_ms;
                    this.messages.unshift(`Saved ${node.node_id}.`);
                })
                .catch(e => {
                    this.messages.unshift(`Error saving ${node.node_id}.`);
                    console.log(e);
                });
        },
        removeMember(member) {
            console.log("Remove", this.getUniqueId(member));

            // Remove from current recommendations
            var index = this.rec_members.indexOf(member);
            if (index > -1) {
                this.rec_members.splice(index, 1);
            }

            // And add to rejected
            this.rejected_members.push(this.getUniqueId(member));
        },
        next() {
            this.$router.push({
                name: "bisect",
                params: { project_id: this.project.project_id }
            });
        },
        keypress(event) {
            if (
                this.node_status != "loaded" ||
                this.rec_status != "loaded" ||
                event.altKey ||
                event.ctrlKey ||
                event.metaKey ||
                event.shiftKey
            ) {
                return;
            }
            if (event.key == "f") {
                this.membersOk();
            } else if (event.key == "j") {
                this.membersNotOk();
            } else if (event.key == "r") {
                this.initialize();
            } else if (event.key == "n" && this.done) {
                console.log("next");
                this.next();
            }
        }
    }
};
</script>

<style>
#bisect-container {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
}

#bisect > * {
    padding: 0 10px;
}

.scrollable {
    margin: 0;
    overflow-y: auto;
}

#decision {
    margin: 0 auto;
}

#decision button {
    margin: 0 1em;
}

/* #messages {
  overflow-y: auto;
  height: 3em;
} */

#node-members .col,
#recommended-members .col {
    padding: 0 5px;
}

#node-members {
    flex: 1;
}

#recommended-members,
#report {
    flex: 2;
}

#progress {
    display: flex;
    flex-wrap: nowrap;
    margin: 0.2em 0;
}

#progress div {
    height: 0.2em;
}

.section-heading {
    margin: 0.2em 0;
}

.spinner-container {
    flex: 2;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.core-component {
    flex: 2;
}
</style>
