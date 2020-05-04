<template>
    <div id="validate">
        <template v-if="node">
            <div id="node-info">
                <v-tooltip>
                    <template v-slot:activator="{ on }">
                        <div v-on="on" class="info-hint mdi mdi-dark mdi-information-outline" />
                    </template>
                    <span>All members of this node, most extreme appearance first.</span>
                </v-tooltip>
                <!--<node-header :node="node" v-if="node" />-->

                <div class="row" v-if="node_members">
                    <div v-for="m of node_members" :key="getUniqueId(m)" class="col col-2">
                        <member-preview
                            :member="m"
                            :controls="member_controls"
                            v-on:moveup="moveupMember"
                        />
                    </div>
                </div>
                <infinite-loading
                    v-if="node"
                    @infinite="updateMembers"
                    spinner="circles"
                    ref="InfiniteLoading"
                >
                    <div slot="no-more" />
                </infinite-loading>
            </div>
            <v-tooltip top>
                <template v-slot:activator="{ on }">
                    <v-progress-linear
                        class="d-block my-2"
                        rounded
                        color="success"
                        :value="project.progress.leaves_n_approved_objects / project.progress.leaves_n_objects * 100"
                        v-on="on"
                    />
                </template>
                <span>{{project.progress.leaves_n_approved_objects}} / {{project.progress.leaves_n_objects}} approved</span>
            </v-tooltip>

            <div id="decision">
                <v-tooltip top>
                    <template v-slot:activator="{ on }">
                        <v-btn
                            id="btn-validate"
                            color="success"
                            @click.prevent="validate(true)"
                            v-on="on"
                        >
                            <i class="mdi mdi-check-all" />
                            <i class="mdi mdi-flag" /> Validate + Flag
                        </v-btn>
                    </template>
                    <span>
                        All members look alike and this cluster is exceptional. Validate and flag for preferred treatment.
                        <kbd>F</kbd>
                    </span>
                </v-tooltip>
                <v-tooltip top>
                    <template v-slot:activator="{ on }">
                        <v-btn
                            id="btn-validate"
                            color="success"
                            @click.prevent="validate(false)"
                            v-on="on"
                        >
                            <i class="mdi mdi-check-all" /> Validate
                        </v-btn>
                    </template>
                    <span>
                        All members look alike. Validate.
                        <kbd>V</kbd>
                    </span>
                </v-tooltip>
                <v-tooltip top>
                    <template v-slot:activator="{ on }">
                        <v-btn id="btn-merge" color="error" @click.prevent="merge" v-on="on">
                            <i class="mdi mdi-call-merge" /> Merge into parent
                        </v-btn>
                    </template>
                    <span>
                        Members are too dissimilar. Merge into parent.
                        <kbd>M</kbd>
                    </span>
                </v-tooltip>
            </div>
        </template>
        <message-log class="bg-light" :messages="messages" />
        <v-dialog persistent v-model="done">
            <v-card>
                <v-card-title>
                    <span class="headline">All nodes are validated in this project.</span>
                </v-card-title>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn
                        v-if="project"
                        variant="primary"
                        :to="{
                        name: 'dataset',
                        params: { dataset_id: project.dataset_id }
                    }"
                    >Back to dataset</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
    </div>
</template>

<script>
// Libraries
import axios from "axios";
import InfiniteLoading from "vue-infinite-loading";

// Internals
import * as api from "@/helpers/api.js";
import mixins from "@/mixins.js";
import globalState from "@/globalState.js";
import exceptions from "@/exceptions.js";

// Components
import MemberPreview from "@/components/MemberPreview.vue";
import MessageLog from "@/components/MessageLog.vue";

export default {
    name: "validate",
    data() {
        return {
            project: null,
            node: null,
            node_members: [],
            members_url: null,
            member_controls: [
                {
                    event: "moveup",
                    icon: "mdi-arrow-up",
                    title: "Move this member to the parent node."
                }
            ],
            error: null,
            done: false,
            view_valid: false
        };
    },
    mounted() {
        console.log("mounted");

        window.addEventListener("keypress", this.keypress);
    },
    beforeDestroy() {
        window.removeEventListener("keypress", this.keypress);
    },
    beforeRouteEnter(to, from, next) {
        // When component is created
        console.log("Validate.beforeRouteEnter");

        next(vm => {
            vm.updateView(to.params.project_id, to.params.node_id);
        });
    },
    beforeRouteUpdate(to, from, next) {
        // When component is updated
        console.log("Validate.beforeRouteUpdate");

        this.updateView(to.params.project_id, to.params.node_id);

        next();
    },
    components: {
        MemberPreview,
        MessageLog,
        InfiniteLoading
    },
    mixins: [mixins],
    methods: {
        setData(project, node, error) {
            this.project = project;
            this.node = node;

            this.done = false;
            if (error instanceof exceptions.NoNextNodeException) {
                this.done = true;
            } else {
                this.error = error;
            }

            if (!error) {
                this.view_valid = true;
            }

            this.node_members = [];
            this.members_url = null;

            // Reset infinite-loading
            if (this.$refs.InfiniteLoading) {
                this.$refs.InfiniteLoading.stateChanger.reset();
            }

            // Update address bar
            var { href } = this.$router.resolve({
                name: "validate",
                params: {
                    project_id: project.project_id,
                    node_id: node.node_id
                }
            });
            window.history.replaceState({}, document.title, href);

            // Update breadcrumb
            this.setBreadcrumbs([
                {
                    text: this.project.dataset.name,
                    to: {
                        name: "dataset",
                        params: { dataset_id: this.project.dataset.dataset_id }
                    },
                    exact: true
                },
                {
                    text: this.project.name,
                    to: {
                        name: "project",
                        params: { project_id: this.project.project_id }
                    },
                    exact: true
                },
                {
                    text: "Validate"
                },
                {
                    text: this.node.name,
                    to: {
                        name: "validate",
                        params: {
                            project_id: this.project.project_id,
                            node_id: this.node.node_id
                        }
                    }
                }
            ]);
        },
        updateView(project_id, node_id) {
            globalState.setLoading("validation");

            // Get project every time to update progress
            var projectPromise =
                (console.log(`Loading project ${project_id}...`),
                api.getProject(project_id, true));

            var nodeIdPromise = node_id
                ? Promise.resolve(node_id)
                : (console.log(`getNextUnapprovedNode ${project_id}...`),
                  api.getNextUnapprovedNode(project_id, null, true));

            var nodePromise = nodeIdPromise.then(node_id => {
                if (node_id == null) {
                    throw exceptions.NoNextNodeException(
                        `No unapproved nodes for project ${project_id}.`
                    );
                }

                console.log(`Loading node ${project_id}::${node_id}...`);
                return api.getNode(project_id, node_id);
            });

            Promise.all([projectPromise, nodePromise])
                .then(([project, node]) => {
                    console.log(project, node);
                    this.setData(project, node, null);
                })
                .catch(e => {
                    console.log(e);
                    this.setData(null, null, e);
                })
                .finally(() => {
                    globalState.unsetLoading("validation");
                });
        },
        // updateMembers gets called as an infinite loading handler.
        updateMembers($state) {
            console.log("updateMembers");

            // Should members_url be updated (with unique id etc.) on response?
            var updateMembersUrl = false;

            if (!this.members_url) {
                const nodes = this.node.children;
                this.members_url = `/api/projects/${
                    this.project.project_id
                }/nodes/${
                    this.node.node_id
                }/members?objects=${!nodes}&nodes=${nodes}&arrange_by=interleaved&`;
                this.page = 0;
                updateMembersUrl = true;
            }

            var url = `${this.members_url}&page=${this.page}`;

            console.log(`Loading ${url}...`);

            axios
                .get(`${this.members_url}&page=${this.page}`)
                .then(response => {
                    this.node_members = this.node_members.concat(
                        response.data.data
                    );

                    if (updateMembersUrl) {
                        this.members_url = response.data.links.self;
                    }

                    $state.loaded();

                    if (this.page < response.data.meta.last_page) {
                        this.page += 1;
                    } else {
                        $state.complete();
                    }
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        validate(preferred = false) {
            console.log("Validate");
            api.patchNode(this.node.project_id, this.node.node_id, {
                validated: true,
                preferred: preferred
            })
                .then(() => {
                    const msg = `Validated node ${this.node.node_id}.`;
                    console.log(msg);
                    this.messages.unshift(msg);

                    this.updateView(this.project.project_id, null);
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        merge() {
            // TODO
            api.mergeNodeInto(
                this.node.project_id,
                this.node.node_id,
                this.node.parent_id
            )
                .then(() => {
                    this.messages.unshift(`Merged ${this.node.node_id}.`);

                    this.updateView(this.project.project_id, null);
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        },
        keypress(event) {
            if (
                !this.view_valid ||
                event.altKey ||
                event.ctrlKey ||
                event.metaKey ||
                event.shiftKey
            ) {
                return;
            }
            if (event.key == "v") {
                this.validate();
            } else if (event.key == "f") {
                this.validate(true);
            } else if (event.key == "m") {
                this.merge();
            }
        },
        moveupMember(member) {
            console.log("Remove", this.getUniqueId(member));

            // TODO: Also reject members.
            api.nodeAdoptMembers(this.node.project_id, this.node.parent_id, [
                member
            ])
                .then(() => {
                    // Remove from current recommendations
                    var index = this.node_members.indexOf(member);
                    if (index > -1) {
                        this.node_members.splice(index, 1);
                    }
                })
                .catch(e => {
                    this.axiosErrorHandler(e);
                });
        }
    }
};
</script>

<style>
#validate {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    flex: 1;
    overflow: hidden;
    height: 100%;
}

#node-info {
    flex: 1;
    overflow-y: auto;
    position: relative;
    padding-top: 0.5em;
}

#node-info .row {
    max-width: 1200px;
    margin: 0 auto;
}

#decision {
    margin: 0 auto;
}

#decision button {
    margin: 0 1em;
}
</style>
