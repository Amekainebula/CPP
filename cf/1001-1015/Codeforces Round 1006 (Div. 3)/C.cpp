//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mpr make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define INF 0x7fffffffffffffff
//#define endl '\n' 
//#define yes cout << "YES"<< endl
//#define no cout << "NO"<< endl
//#define yess cout << "Yes" << endl
//#define noo cout << "No"<< endl
//using namespace std;
////int ans[2100005];
//bool check(int n, int k)
//{
//    return(n >> k) & 1;
//}
//void solve()
//{
//    int n, k;
//    cin >> n >> k;
//    int cnt = 0;
//    int temp = -1;
//    int kk = k;
//    string s;
//
//    while (kk > 0)
//    {
//        cnt++;
//        if(kk%2==0)
//        {
//            s += "0";
//            temp = cnt;
//            break;
//        }
//        else s+="1";
//        kk/=2;
//    }   
//    //cout<<s<<endl;
//    
//    if (temp == 1)
//    {
//        cout << k << " ";
//        for (int i = 0; i < n - 1; i++)
//            cout << "0 ";
//        cout << endl;
//        return;
//    }
//    if (k == (1 << cnt) - 1)
//    {
//        if (n <= (k + 1) / 2)
//        {
//            //cout << (k + 1) / 2 << " ";
//            cout << k << " ";
//            for (int i = 0; i < n - 1; i++)
//            {
//                if (i <= k)
//                    cout << i << " ";
//                else
//                    cout << "0 ";
//            }
//        }
//        else
//        {
//            for (int i = 0; i < n; i++)
//            {
//                if(i <= k)
//                    cout << i << " ";
//                else
//                    cout << "0 ";
//            }
//        }
//        cout << endl;
//        return;
//    }
//    else
//    {
//        int tt = 1 << (temp - 1);
//        /*if (n <= tt - 1)
//        {*/
//            cout << k << " ";
//            for (int i = 0; i < n - 1; i++)
//            {
//                if (i < tt)
//                    cout << i << " ";
//                else
//                    cout << "0 ";
//            }
//            cout << endl;
//            return;
//       /* }
//        else
//        {
//
//        }*/
//    }
//    //int cnt = 0;
//    ///*while (k > 0)
//    //{
//    //    int j = 1;
//    //    while (j < k)
//    //        j <<= 1;
//    //    k -= j;
//    //    cnt += j;
//    //}*/
//    //int j = 1;
//    //while (j < k)
//    //    j <<= 1;
//    //if (j != k)j >>= 1;
//    //cnt += j;
//    //k -= j;
//    //if (cnt + 1 <= n)
//    //{
//    //    for (int i = 0; i <= cnt; i++)
//    //        cout<<ans[i]<<" ";
//    //    for (int i = n - cnt - 1; i > 0; i--)
//    //        cout << "0 ";
//    //    cout << endl;
//    //    return;
//    //}
//    //else
//    //{
//    //    cout << cnt<< " ";
//    //    for (int i = 0; i < n - 1; i++)
//    //        cout << ans[i] << " ";
//    //    cout << endl;
//    //}
//}
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//   
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}