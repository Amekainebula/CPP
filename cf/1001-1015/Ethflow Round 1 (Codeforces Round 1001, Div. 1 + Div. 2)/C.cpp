//#define _CRT_SECURE_NO_WARNINGS 1
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
//using namespace std;
//int a[100005];
//int temp[100005];
//int v[100005];
//void solve()
//{
//    //int maxx = -2000;
//    int sum = 0;
//    int n;
//    cin >> n;
//    for (int i = 1; i <= n; i++)
//    {
//        cin >> a[i];
//        sum += a[i];
//        /*if (a[i] > maxx)
//        {
//            maxx = a[i];
//        }*/
//    }
//    if (n == 1)
//    {
//        cout << a[1] << endl;
//        return;
//    }
//    int cnt = 0;
//    int ssum = 0;
//    for (int i = 1; i <= n - 1; i++)
//    {
//        cnt++;
//        v[cnt] = a[i + 1] - a[i];
//        ssum += v[cnt];
//    }
//    int cntt = cnt;
//    int ans = max(abs(ssum), sum);
//    while (cnt > 1)
//    {
//        int summ = 0;
//        if (cntt % 2 == cnt % 2)
//        {
//            for (int i = 1; i < cnt; i++)
//            {
//                temp[i] = v[i + 1] - v[i];
//                summ += temp[i];
//            }
//        }
//        else
//        {
//            for (int i = 1; i < cnt; i++)
//            {
//                v[i] = temp[i + 1] - temp[i];
//                summ += v[i];
//            }
//        }
//        ans = max(ans, abs(summ));
//        cnt--;
//    }
//    /*if (n == 2)
//    {
//        cout << max(max(abs(v[1]), maxx), sum) << endl;
//        return;
//    }
//    if (cntt % 2 == 0)
//    {
//        cout << max(abs(temp[1]), maxx) << endl;
//    }
//    else
//    {
//        cout << max(abs(v[1]), maxx) << endl;
//    }*/
//    cout << ans << endl;
//    //cout << sum1 << ' ' << sum2 << endl;
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