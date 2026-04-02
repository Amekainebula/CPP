#include <iostream>
#include <algorithm>
#include <vector>
#define int long long
using namespace std;
// 计算满足去除对应两个元素后剩余元素之和小于x的数对数量
int calcLessThanX(vector<int>a, int x)
{
    int n = a.size();
    int s = 0;
    for (int num : a) 
    {
        s += num;
    }
    int j = 0;
    int ans = 0;
    // 逆向遍历i
    for (int i = n - 1; i >= 0; --i) 
    {
        while (j < n && s - a[i] - a[j] >= x)
        {
            j++;
        }
        ans += (n - j);
    }
    // 处理i = j的情况
    for (int i = 0; i < n; ++i) 
    {
        if (s - 2 * a[i] < x)
        {
            ans -= 1;
        }
    }
    return ans / 2;
}

signed main() 
{
    int t;
    cin >> t;
    while (t--) 
    {
        int n, x, y;
        cin >> n >> x >> y;
        vector<int> a(n);
        for (int i = 0; i < n; ++i) 
        {
            cin >> a[i];
        }
        sort(a.begin(), a.end());
        cout << calcLessThanX(a, y + 1) - calcLessThanX(a, x) << endl;
    }
    return 0;
}
//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define double long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//int t, a[200005];
//int aa[200005];
//signed main()
//{
//    cin >> t;
//    while (t--)
//    {
//        int n, x, y;
//        int sum = 0;
//        int ans = 0;
//        cin >> n >> x >> y;
//        for (int i = 1; i <= n; i++)
//        {
//            cin >> a[i];
//            aa[i] = a[i];
//            sum += a[i];
//        }
//        sort(a + 1, a + n + 1);
//        int tds = 1;
//        for (int i = 1; i <= n; i++)
//        {
//            int maxx = sum - x - aa[i];
//            int minn = sum - y - aa[i];
//            for (int j = 1; j <= n; j++)
//            {
//                if (a[j] > maxx)break;
//                if (a[j] >= minn && a[j] <= maxx)
//                {
//                    if (a[j] == aa[i] && tds == 1)
//                    {
//                        tds = 0;
//                        continue;
//                    }
//                    ans++;
//                }
//            }
//        }
//        cout << ans << endl;
//    }
//    return 0;
//}