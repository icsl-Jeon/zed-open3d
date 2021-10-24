//
// Created by jbs on 21. 9. 6..
//

#include <Misc.h>
namespace  misc {



    int DBSCAN::run() {
        int clusterID = 1;
        vector<DepthPoint>::iterator iter;
        for (iter = m_points.begin(); iter != m_points.end(); ++iter) {
            if (iter->clusterID == UNCLASSIFIED) {
                if (expandCluster(*iter, clusterID) != FAILURE) {
                    clusterID += 1;
                }
            }
        }
        n_totalCluster = clusterID;
        return 0;
    }

    int DBSCAN::getNNoise() {
        int nNoise = 0;
        for (const auto& pnt: m_points )
            nNoise += int ( pnt.clusterID == UNCLASSIFIED);
        return nNoise;
    }

    int DBSCAN::expandCluster(DepthPoint point, int clusterID) {
        vector<int> clusterSeeds = calculateCluster(point);

        if (clusterSeeds.size() < m_minPoints) {
            point.clusterID = NOISE;
            return FAILURE;
        } else {
            int index = 0, indexCorePoint = 0;
            vector<int>::iterator iterSeeds;
            for (iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds) {
                m_points.at(*iterSeeds).clusterID = clusterID;
                if (m_points.at(*iterSeeds).depth == point.depth ) {
                    indexCorePoint = index;
                }
                ++index;
            }
            clusterSeeds.erase(clusterSeeds.begin() + indexCorePoint);

            for (vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i) {
                vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

                if (clusterNeighors.size() >= m_minPoints) {
                    vector<int>::iterator iterNeighors;
                    for (iterNeighors = clusterNeighors.begin();
                         iterNeighors != clusterNeighors.end(); ++iterNeighors) {
                        if (m_points.at(*iterNeighors).clusterID == UNCLASSIFIED ||
                            m_points.at(*iterNeighors).clusterID == NOISE) {
                            if (m_points.at(*iterNeighors).clusterID == UNCLASSIFIED) {
                                clusterSeeds.push_back(*iterNeighors);
                                n = clusterSeeds.size();
                            }
                            m_points.at(*iterNeighors).clusterID = clusterID;
                        }
                    }
                }
            }

            return CLUSTER_SUCCESS;
        }
    }

    vector<int> DBSCAN::calculateCluster(DepthPoint point) {
        int index = 0;
        vector<DepthPoint>::iterator iter;
        vector<int> clusterIndex;
        for (iter = m_points.begin(); iter != m_points.end(); ++iter) {
            if (calculateDistance(point, *iter) <= m_epsilon) {
                clusterIndex.push_back(index);
            }
            index++;
        }
        return clusterIndex;
    }

    inline double DBSCAN::calculateDistance(const DepthPoint &pointCore, const DepthPoint &pointTarget) {
        return pow(pointCore.depth - pointTarget.depth, 2) ;
    }

}