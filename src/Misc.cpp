//
// Created by jbs on 21. 9. 6..
//

#include <Misc.h>
namespace  misc {


    void drawPred(string className, float conf, float x, float y, float z ,int left, int top, int right, int bottom, cv::Mat &frame) {
        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0),2);

        std::string label = cv::format("%.2f", conf);
        label = className + ": " + label;
        std::string position = cv::format("[%.2f,%.2f,%.2f]",x,y,z);

        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        top = max(top, labelSize.height);
        rectangle(frame, cv::Point(left, top - labelSize.height),
                  cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
        putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());

        putText(frame, position, cv::Point(left,bottom + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255));
    }

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