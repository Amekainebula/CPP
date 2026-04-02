//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mp make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define endl '\n'
//using namespace std;
//int big[100005];
//int small[100005];
//map<int, int> mpp;
//void solve()
//{
//    mpp.clear();
//    int ans = 0;
//    int n, k;
//    cin >> n >> k;
//    int cntb = 0, cnts = 0;
//    for (int i = 1; i <= k; i++)
//        mpp[i] = 0;
//    for (int i = 1; i <= n; i++)
//    {
//        int x;
//        cin >> x;
//        if (x >= k)
//        {
//            cntb++;
//            big[cntb] = x;
//        }
//        else
//        {
//            cnts++;
//            small[cnts] = x;
//            mpp[x]++; 
//        }
//    }
//    if (cnts < 2)
//    {
//        cout << "0" << endl;
//        return;
//    }
//    sort(small + 1, small + cnts + 1);
//    for (int i = 1; i <= k / 2; i++)
//    {
//        int temp = mpp[k - i];
//        if (i * 2 != k)
//        {
//            if (mpp[i] > 0 && temp > 0)
//            {
//                ans += min(mpp[i], temp);
//                mpp[k - i] -= min(mpp[i], temp);
//                mpp[i] -= min(mpp[i], temp);
//            }
//        }
//        else
//        {
//            if (mpp [i]> 1)
//            {
//                ans += mpp[i] / 2;
//                mpp[i] %= 2;
//            }
//        }
//    }
//    
//    cout << ans << endl;
//}
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}